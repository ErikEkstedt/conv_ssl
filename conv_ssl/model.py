import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pytorch_lightning as pl
from torchmetrics import Metric

from conv_ssl.models import EncoderPretrained, AR
from conv_ssl.utils import OmegaConfArgs, repo_root, load_config

from datasets_turntaking.features.vad import VadProjection


class ShiftHoldMetric(Metric):
    """Used in conjuction with 'VadProjection' from datasets_turntaking"""

    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("hold_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("hold_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("shift_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("shift_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        hold_correct: torch.Tensor,
        hold_total: torch.Tensor,
        shift_correct: torch.Tensor,
        shift_total: torch.Tensor,
    ):
        self.hold_correct += hold_correct
        self.hold_total += hold_total
        self.shift_correct += shift_correct
        self.shift_total += shift_total

    def stats(self, ac, an, bc, bn):
        """
        F1 statistics over Shift/Hold

        Example 'Shift':
            * ac = shift_correct
            * an = shift_total
            * bc = hold_correct
            * bn = hold_total

        True Positives:  shift_correct
        False Negatives:  All HOLD predictions at SHIFT locations -> (shift_total - shift_correct)
        True Negatives:  All HOLD predictions at HOLD locations -> hold_correct
        False Positives:  All SHIFT predictions at HOLD locations -> (hold_total - hold_correct)

        Symmetrically true for Holds.
        """
        EPS = 1e-9
        tp = ac
        fn = an - ac
        tn = bc
        fp = bn - bc
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        support = tp + fn
        f1 = tp / (tp + 0.5 * (fp + fn) + EPS)
        return {
            "f1": f1,
            "support": support,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    def compute(self):
        """Compute final result"""
        stats = {
            "hold": self.stats(
                ac=self.hold_correct,
                an=self.hold_total,
                bc=self.shift_correct,
                bn=self.shift_total,
            ),
            "shift": self.stats(
                ac=self.shift_correct,
                an=self.shift_total,
                bc=self.hold_correct,
                bn=self.hold_total,
            ),
        }

        # Weighted F1 score
        # scaled/weighted by the support of each metric
        # shift_f1*shift_support + hold_f1*hold_support )/ (shift_support + hold_support)
        f1h = stats["hold"]["f1"] * stats["hold"]["support"]
        f1s = stats["shift"]["f1"] * stats["shift"]["support"]
        tot = stats["hold"]["support"] + stats["shift"]["support"]
        stats["f1_weighted"] = (f1h + f1s) / tot
        return stats


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
        # self.vad_projection = nn.Linear(input_dim, conf["vad_class_prediction"])
        if conf["vad_class_prediction"]["regression"]:
            cf = len(conf["vad_class_prediction"]["bin_times"]) * 2
            self.projection_head = nn.Sequential(
                nn.Linear(input_dim, cf),
                Rearrange("... (c f) -> ... c f", c=2, f=cf // 2),
            )
        else:
            self.projection_head = nn.Linear(
                input_dim, conf["vad_class_prediction"]["n_classes"]
            )

    def build_encoder(self, conf):
        encoder = EncoderPretrained(conf)
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
            z += vc[:, : z.shape[1]]

        # logits_ar & logits_vp
        if self.ulm is not None:
            z = self.ulm(z)["z"]
            out["logits_ar"] = self.ulm_head(z)

        if self.conf["vad_cond"]["post_ulm"]:
            z += vc[:, : z.shape[1]]

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
        self.val_metric = ShiftHoldMetric()
        self.vad_projection = VadProjection(
            bin_times=conf["vad_class_prediction"]["bin_times"],
            vad_threshold=conf["vad_class_prediction"]["vad_threshold"],
            pred_threshold=conf["vad_class_prediction"]["pred_threshold"],
            event_min_context=conf["vad_class_prediction"]["event_min_context"],
            event_min_duration=conf["vad_class_prediction"]["event_min_duration"],
            event_horizon=conf["vad_class_prediction"]["event_horizon"],
            event_start_pad=conf["vad_class_prediction"]["event_start_pad"],
            event_target_duration=conf["vad_class_prediction"]["event_target_duration"],
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
        s = "VPN\n"
        s += "Encoder\n"
        s += f"\ttype: {self.conf['encoder']['type']}\n"
        s += f"\toutput_layer: {self.conf['encoder']['output_layer']}\n"
        s += f"\tHz: {self.conf['encoder']['frame_hz']}\n"
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
        s += f"\tregression: {self.conf['vad_class_prediction']['regression']}\n"
        s += f"\thead: {self.net.projection_head}\n"
        return s

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def calc_losses(self, out, vad_labels, input_ids=None, reduction="mean"):
        loss = {}

        # Vad Projection Loss
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
            batch_size: int, size of batch
        """

        out = self(
            waveform=batch.get("waveform", None),
            input_ids=batch.get("q", None),
            vad=batch["vad"],
            vad_history=batch.get("vad_history", None),
        )

        n_frames = out["logits_vp"].shape[1]
        # Resize batch/use-subset to match encoder output size
        # some encoders (wav2vec, vq_wav2vec) drops 2 frames on 10sec audio
        # some encoders (wavlm_base, hubert) drops 1 frames (after downsampling) on 10sec audio
        batch["vad"] = batch["vad"][:, :n_frames]
        batch["vad_label"] = batch["vad_label"][:, :n_frames]
        if "vad_history" in batch:
            batch["vad_history"] = batch["vad_history"][:, :n_frames]

        loss = self.calc_losses(
            out,
            vad_labels=batch["vad_label"],
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

        m = self.vad_projection(out["logits_vp"], batch["vad"])
        self.val_metric.update(
            hold_correct=m["hold"]["correct"],
            hold_total=m["hold"]["n"],
            shift_correct=m["shift"]["correct"],
            shift_total=m["shift"]["n"],
        )

    def validation_epoch_end(self, outputs) -> None:
        # self.log('val/hold_shift', self.val_metric)
        r = self.val_metric.compute()
        self.log("val_f1_weighted", r["f1_weighted"])
        self.log("val_f1_shift", r["shift"]["f1"])
        self.val_metric.reset()

    def test_step(self, batch, batch_idx, **kwargs):
        loss, out, batch = self.shared_step(batch)
        batch_size = batch["vad"].shape[0]

        self.log("test_loss", loss["total"], batch_size=batch_size)
        self.log("test_loss_vp", loss["vp"], batch_size=batch_size)

        if "loss_ar" in loss:
            self.log("test_loss_ar", loss["ar"], batch_size=batch_size)

        if not getattr(self, "test_metric"):
            self.test_metric = ShiftHoldMetric()

        m = self.vad_projection(out["logits_vp"], batch["vad"])
        self.test_metric.update(
            hold_correct=m["hold"]["correct"],
            hold_total=m["hold"]["n"],
            shift_correct=m["shift"]["correct"],
            shift_total=m["shift"]["n"],
        )

    def test_epoch_end(self, outputs) -> None:
        r = self.test_metric.compute()
        self.log("test/f1_weighted", r["f1_weighted"])
        self.log("test/f1_shift", r["shift"]["f1"])
        self.test_metric.reset()
        # self.log('val/hold_shift', self.val_metric)

    @property
    def run_name(self):
        # Name the run e.g. hubert_44_41
        name = self.conf["encoder"]["type"].replace("_base", "")
        name += f"_{self.conf['ulm']['num_layers']}{self.conf['ulm']['num_heads']}"
        name += f"_{self.conf['ar']['num_layers']}{self.conf['ar']['num_heads']}"
        if self.conf["vad_class_prediction"]["regression"]:
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
    model = VPModel(conf)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params}")

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
