import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pytorch_lightning as pl

from conv_ssl.models import Encoder, AR
from conv_ssl.utils import OmegaConfArgs, repo_root, load_config

from vad_turn_taking.metrics import ShiftHoldMetric
from vad_turn_taking.vad_projection import VadLabel, ProjectionCodebook


class VadCondition(nn.Module):
    def __init__(self, dim, vad_history=False, vad_history_bins=5) -> None:
        super().__init__()
        self.dim = dim

        # Vad Condition
        # vad: 2 one-hot encodings
        # self.vad_condition = nn.Embedding(num_embeddings=2, embedding_dim=dim)
        self.vad_condition = nn.Linear(2, dim)

        if vad_history:
            self.vad_history = nn.Linear(vad_history_bins, dim)

        self.ln = nn.LayerNorm(dim)
        # self.init()

    def init(self):
        # init orthogonal vad vectors
        nn.init.orthogonal_(self.vad_condition.weight.data)

    def forward(self, vad, vad_history=None):
        v_cond = self.vad_condition(vad)

        # Add vad-history information
        if vad_history is not None:
            v_cond += self.vad_history(vad_history)

        return self.ln(v_cond)


class ProjectionModel(nn.Module):
    @staticmethod
    def load_config(path=None, args=None, format="dict"):
        if path is None:
            path = repo_root() + "/conv_ssl/config/model.yaml"
        return load_config(path, args=args, format=format)

    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf

        # Audio Encoder
        self.encoder = self.build_encoder(conf)
        input_dim = conf["encoder"]["dim"]

        # VAD Conditioning
        self.vad_condition = VadCondition(
            input_dim,
            vad_history=conf["vad_cond"]["vad_history"],
            vad_history_bins=conf["vad_cond"]["vad_history_bins"],
        )

        # Unit Language Model
        self.ulm = self.build_ulm(input_dim, conf)
        if self.ulm is not None:
            self.ulm_head = nn.Linear(conf["ulm"]["dim"], conf["quantizer"]["n_codes"])
            input_dim = conf["ulm"]["dim"]

        # Autoregressive
        self.ar = self.build_ar(input_dim, conf)
        if self.ar is not None:
            input_dim = conf["ar"]["dim"]

        # VAD Projection
        # self.vad_projection = nn.Linear(input_dim, conf["vad_projection"])
        if conf["vad_projection"]["regression"]:
            cf = len(conf["vad_projection"]["bin_times"]) * 2
            self.projection_head = nn.Sequential(
                nn.Linear(input_dim, cf),
                Rearrange("... (c f) -> ... c f", c=2, f=cf // 2),
            )
        else:
            total_bins = 2 * len(conf["vad_projection"]["bin_times"])
            self.n_classes = 2 ** total_bins
            self.projection_head = nn.Linear(input_dim, self.n_classes)

    def build_encoder(self, conf):
        encoder = Encoder(conf)
        return encoder

    def build_ulm(self, input_dim, conf):
        net = None
        if conf["quantizer"]["n_codes"] > 0 and conf["ulm"]["num_layers"] > 0:
            net = AR(
                input_dim=input_dim,
                dim=conf["ulm"]["dim"],
                num_layers=conf["ulm"]["num_layers"],
                dropout=conf["ulm"]["dropout"],
                ar=conf["ulm"]["type"],
                transfomer_kwargs=dict(
                    num_heads=conf["ulm"]["num_heads"],
                    dff_k=conf["ulm"]["dff_k"],
                    use_pos_emb=conf["ulm"]["use_pos_emb"],
                    abspos=conf["ulm"]["abspos"],
                    sizeSeq=conf["ulm"]["sizeSeq"],
                ),
            )
        return net

    def build_ar(self, input_dim, conf):
        net = None
        if conf["ar"]["num_layers"] > 0:
            net = AR(
                input_dim=input_dim,
                dim=conf["ar"]["dim"],
                num_layers=conf["ar"]["num_layers"],
                dropout=conf["ar"]["dropout"],
                ar=conf["ar"]["type"],
                transfomer_kwargs=dict(
                    num_heads=conf["ar"]["num_heads"],
                    dff_k=conf["ar"]["dff_k"],
                    use_pos_emb=conf["ar"]["use_pos_emb"],
                    abspos=conf["ar"]["abspos"],
                    sizeSeq=conf["ar"]["sizeSeq"],
                ),
            )
        return net

    def loss_vad_projection(self, logits, labels, reduction="mean"):
        # CrossEntropyLoss over discrete labels
        loss = F.cross_entropy(
            einops.rearrange(logits, "b n d -> (b n) d"),
            einops.rearrange(labels, "b n -> (b n)"),
            reduction=reduction,
        )
        if reduction == "none":
            n = logits.shape[1]
            loss = einops.rearrange(loss, "(b n) -> b n", n=n)
        return loss

    def loss_ulm(self, logits_ar, input_ids, reduction="mean"):
        """
        Calculate ULM unit autoregrssive loss (Causal Language Modeling)
        """
        # Shift the input to get y/targets/labels
        y = input_ids[:, 1:]
        y_hat = logits_ar[:, :-1]
        loss_ar = F.cross_entropy(
            einops.rearrange(y_hat, "b n d -> (b n) d"),
            einops.rearrange(y, "b n -> (b n)"),
            reduction=reduction,
        )

        if reduction == "none":
            n = y.shape[1]  # steps in loss
            loss_ar = einops.rearrange(loss_ar, "(b n) -> b n", n=n)
        return loss_ar

    def encode(self, waveform, input_ids):
        if input_ids is not None:
            return input_ids
        else:
            enc_out = self.encoder(waveform)
            z = enc_out.get("q_idx", enc_out["z"])
            return z

    def forward(self, waveform=None, input_ids=None, vad=None, vad_history=None):
        assert (
            waveform is not None or input_ids is not None
        ), "must provide either `waveform` or `input_ids`"

        out = {}

        # Encode Audio
        z = self.encode(waveform, input_ids)

        # Vad conditioning
        vc = self.vad_condition(vad, vad_history)

        if self.conf["vad_cond"]["pre_ulm"]:
            # z += vc[:, : z.shape[1]] # don't do inplace! deterministic/cuda fails
            z = z + vc[:, : z.shape[1]]

        # logits_ar & logits_vp
        if self.ulm is not None:
            z = self.ulm(z)["z"]
            out["logits_ar"] = self.ulm_head(z)

        if self.conf["vad_cond"]["post_ulm"]:
            # z += vc[:, : z.shape[1]]
            z = z + vc[:, : z.shape[1]]

        if self.ar is not None:
            z = self.ar(z)["z"]

        out["logits_vp"] = self.projection_head(z)

        return out


class VPModel(pl.LightningModule):
    @staticmethod
    def load_config(path=None, args=None, format="dict"):
        return ProjectionModel.load_config(path, args, format)

    def __init__(self, conf) -> None:
        super().__init__()
        self.net = ProjectionModel(conf)

        # Metrics
        self.val_metric = ShiftHoldMetric(
            min_context=conf["vad_projection"]["event_min_context"],
            horizon=conf["vad_projection"]["event_horizon"],
            start_pad=conf["vad_projection"]["event_start_pad"],
            target_duration=conf["vad_projection"]["event_target_duration"],
            frame_hz=self.net.encoder.frame_hz,
        )
        self.test_metric = None  # set in test if necessary
        self.vad_label_maker = VadLabel(
            bin_times=conf["vad_projection"]["bin_times"],
            vad_hz=self.net.encoder.frame_hz,
            threshold_ratio=conf["vad_projection"]["vad_threshold"],
        )
        self.event_dict = dict(
            bin_times=conf["vad_projection"]["bin_times"],
            vad_threshold=conf["vad_projection"]["vad_threshold"],
            pred_threshold=conf["vad_projection"]["pred_threshold"],
            event_min_context=conf["vad_projection"]["event_min_context"],
            event_min_duration=conf["vad_projection"]["event_min_duration"],
            event_horizon=conf["vad_projection"]["event_horizon"],
            event_start_pad=conf["vad_projection"]["event_start_pad"],
            event_target_duration=conf["vad_projection"]["event_target_duration"],
            frame_hz=self.net.encoder.frame_hz,
        )

        # only use discrete codes if necessary
        self.regression = conf["vad_projection"]["regression"]
        if not self.regression:
            self.projection_codebook = ProjectionCodebook(
                bin_times=conf["vad_projection"]["bin_times"],
                frame_hz=self.net.encoder.frame_hz,
            )

        # conf
        self.frame_hz = self.net.encoder.frame_hz
        self.conf = conf

        # Training params
        self.learning_rate = conf["optimizer"]["learning_rate"]
        self.betas = conf["optimizer"]["betas"]
        self.alpha = conf["optimizer"]["alpha"]
        self.weight_decay = conf["optimizer"]["weight_decay"]
        self.save_hyperparameters()

    def summary(self):
        s = "VPModel\n"
        s += "Encoder\n"
        s += f"\ttype: {self.conf['encoder']['type']}\n"
        s += f"\toutput_layer: {self.conf['encoder']['output_layer']}\n"
        s += f"\tHz: {self.net.encoder.frame_hz}\n"
        s += f"\tquantizer n_codes: {self.conf['quantizer']['n_codes']}\n"
        if self.net.ulm is not None:
            s += "ULM\n"
            s += f"\tnum_layers: {self.conf['ulm']['num_layers']}\n"
            s += f"\tnum_heads: {self.conf['ulm']['num_heads']}\n"
            s += f"\tdim: {self.conf['ulm']['dim']}\n"
        if self.net.ar is not None:
            s += "AR\n"
            s += f"\tnum_layers: {self.conf['ar']['num_layers']}\n"
            s += f"\tnum_heads: {self.conf['ar']['num_heads']}\n"
            s += f"\tdim: {self.conf['ar']['dim']}\n"
        s += "Head\n"
        s += f"\tregression: {self.regression}\n"
        s += f"\thead: {self.net.projection_head}\n"
        s += f"\tbin_times: {self.conf['vad_projection']['bin_times']}\n"
        if self.regression:
            s += f"\nregression_loss: {self.conf['vad_projection']['regression_loss']}"
        return s

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

    def get_next_speaker_probs(self, logits, vad=None):
        if self.regression:
            probs = logits.sigmoid()
            next_probs = probs.mean(dim=-1)
            tot = next_probs.sum(dim=-1)
            # renormalize for comparison
            next_probs[..., 0] = next_probs[..., 0] / tot
            next_probs[..., 1] = next_probs[..., 1] / tot
        else:
            next_probs = self.projection_codebook.get_next_speaker_probs(logits, vad)
        return next_probs

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def calc_losses(self, out, vad_projection_window, input_ids=None, reduction="mean"):
        loss = {}

        # Vad Projection Loss
        if self.conf["vad_projection"]["regression"]:
            if self.conf["vad_projection"]["regression_loss"] == "mse":
                loss["vp"] = F.mse_loss(
                    out["logits_vp"].sigmoid(), vad_projection_window
                )
            elif self.conf["vad_projection"]["regression_loss"] in ["mae", "l1"]:
                loss["vp"] = F.l1_loss(
                    out["logits_vp"].sigmoid(), vad_projection_window
                )
            else:
                loss["vp"] = F.binary_cross_entropy_with_logits(
                    out["logits_vp"], vad_projection_window
                )
        else:
            # Vad Projection Loss
            vad_labels = self.projection_codebook(vad_projection_window)
            loss["vp"] = self.net.loss_vad_projection(
                logits=out["logits_vp"], labels=vad_labels, reduction=reduction
            )

        # ULM Loss
        if self.net.ulm is not None and input_ids is not None:
            loss["ar"] = self.net.loss_ulm(
                out["logits_ar"], input_ids, reduction=reduction
            )

            if reduction == "none":
                loss["total"] = (
                    self.alpha * loss["vp"][:, :-1] + (1 - self.alpha) * loss["ar"]
                )
            else:
                loss["total"] = self.alpha * loss["vp"] + (1 - self.alpha) * loss["ar"]
        else:
            loss["total"] = loss["vp"]

        return loss

    def shared_step(self, batch, reduction="mean"):
        """
        Arguments:
            batch:      dict, containing 'waveform' or 'q' (input_ids), vad, vad_history

        Returns:
            loss:       torch.Tensor
            out:        dict
            batch:      same as input arguments (fixed for differenct encoder Hz)
        """
        # Extract labels (using horizon which spans beyond the sample)
        vad_projection_window = self.vad_label_maker.vad_projection(batch["vad"])

        # Only keep the relevant vad information
        n_valid = vad_projection_window.shape[1]
        batch["vad"] = batch["vad"][:, :n_valid]

        # Forward pass
        out = self(
            waveform=batch.get("waveform", None),
            input_ids=batch.get("q", None),
            vad=batch["vad"],
            vad_history=batch.get("vad_history", None),
        )

        # Resize batch/use-subset to match encoder output size
        # some encoders (wav2vec, vq_wav2vec) drops 2 frames on 10sec audio
        # some encoders (wavlm_base, hubert) drops 1 frames (after downsampling) on 10sec audio
        n_frames = out["logits_vp"].shape[1]
        batch["vad"] = batch["vad"][:, :n_frames]
        batch["vad_projection_window"] = vad_projection_window[:, :n_frames]
        if "vad_history" in batch:
            batch["vad_history"] = batch["vad_history"][:, :n_frames]

        loss = self.calc_losses(
            out,
            vad_projection_window=batch["vad_projection_window"],
            input_ids=batch.get("q", None),
            reduction=reduction,
        )

        return loss, out, batch

    def training_step(self, batch, batch_idx, **kwargs):
        loss, _, _ = self.shared_step(batch)
        batch_size = batch["vad"].shape[0]

        self.log("loss", loss["total"], batch_size=batch_size)
        self.log("loss_vp", loss["vp"], batch_size=batch_size)

        if "loss_ar" in loss:
            self.log("loss_ar", loss["ar"], batch_size=batch_size)

        return {"loss": loss["total"]}

    def validation_step(self, batch, batch_idx, **kwargs):
        loss, out, batch = self.shared_step(batch)
        batch_size = batch["vad"].shape[0]

        self.log("val_loss", loss["total"], batch_size=batch_size)
        self.log("val_loss_vp", loss["vp"], batch_size=batch_size)
        if "loss_ar" in loss:
            self.log("val_loss_ar", loss["ar"], batch_size=batch_size)

        next_probs = self.get_next_speaker_probs(out["logits_vp"], vad=batch["vad"])
        self.val_metric.update(next_probs, vad=batch["vad"])

    def validation_epoch_end(self, outputs) -> None:
        r = self.val_metric.compute()
        for metric_name, values in r.items():
            if isinstance(values, dict):
                for val_name, val in values.items():
                    if val_name in ["tp", "tn", "fp", "fn"]:
                        continue
                    if "support" in val_name:
                        continue

                    self.log(f"val_{metric_name}_{val_name}", val)
            else:
                self.log(f"val_{metric_name}", values)
        self.val_metric.reset()

    def test_step(self, batch, batch_idx, **kwargs):
        loss, out, batch = self.shared_step(batch)
        batch_size = batch["vad"].shape[0]

        self.log("test_loss", loss["total"], batch_size=batch_size)
        self.log("test_loss_vp", loss["vp"], batch_size=batch_size)
        if "loss_ar" in loss:
            self.log("test_loss_ar", loss["ar"], batch_size=batch_size)

        if self.test_metric is None:
            self.test_metric = ShiftHoldMetric()

        next_probs = self.get_next_speaker_probs(out["logits_vp"], vad=batch["vad"])
        self.test_metric.update(next_probs, vad=batch["vad"])

    def test_epoch_end(self, outputs) -> None:
        r = self.test_metric.compute()
        for metric_name, values in r.items():
            if isinstance(values, dict):
                for val_name, val in values.items():
                    if not val_name in ["tp", "tn", "fp", "fn"]:
                        self.log(f"test/{metric_name}_{val_name}", val)
            else:
                self.log(f"test/{metric_name}", values)
        self.test_metric.reset()

    @property
    def run_name(self):
        # Name the run e.g. hubert_44_41
        name = self.conf["encoder"]["type"].replace("_base", "")
        name += f"_{self.conf['ulm']['num_layers']}{self.conf['ulm']['num_heads']}"
        name += f"_{self.conf['ar']['num_layers']}{self.conf['ar']['num_heads']}"
        if self.conf["vad_projection"]["regression"]:
            name += "_reg"
        return name

    @staticmethod
    def add_model_specific_args(parent_parser):
        """argparse arguments for SoSIModel (based on yaml-config)"""
        parser = parent_parser.add_argument_group("VPModel")
        parser.add_argument("--conf", default=None, type=str)
        parser.add_argument("--dont_log_model", action="store_true")

        # A workaround for OmegaConf + WandB-Sweeps
        default_conf = VPModel.load_config(format=None)
        parser = OmegaConfArgs.add_argparse_args(parser, default_conf)
        return parent_parser


if __name__ == "__main__":
    from argparse import ArgumentParser
    from datasets_turntaking import DialogAudioDM
    from conv_ssl.utils import count_parameters

    parser = ArgumentParser()
    parser = DialogAudioDM.add_data_specific_args(parser)
    # parser = VPModel.add_model_specific_args(parser)
    args = parser.parse_args()
    data_conf = DialogAudioDM.load_config(path=args.data_conf, args=args)
    # data_conf["dataset"]["type"] = "sliding"
    DialogAudioDM.print_dm(data_conf, args)

    # Model
    conf = VPModel.load_config()
    conf["vad_projection"]["regression"] = False
    conf["vad_projection"][
        "regression_loss"
    ] = "bce"  # 'mae', 'l1', 'mse' otherwise 'bce'
    model = VPModel(conf)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params}")
    cuda = True
    if cuda:
        model = model.to("cuda")

    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        audio_context_duration=data_conf["dataset"]["audio_context_duration"],
        ipu_min_time=data_conf["dataset"]["ipu_min_time"],
        ipu_pause_time=data_conf["dataset"]["ipu_pause_time"],
        vad_hz=model.frame_hz,
        vad_bin_times=data_conf["dataset"]["vad_bin_times"],
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=4,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup()
    diter = iter(dm.val_dataloader())

    batch = next(diter)
    if cuda:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to("cuda")
    loss, out, batch = model.shared_step(batch)
