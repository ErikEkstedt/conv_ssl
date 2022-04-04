import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pytorch_lightning as pl

from conv_ssl.models import Encoder, AR
from conv_ssl.utils import OmegaConfArgs, repo_root, read_json, load_config, to_device
from vap_turn_taking import VAP, TurnTakingMetrics


d = read_json("conv_ssl/config/event_settings.json")
METRIC_CONF = d["metric"]
HS_CONF = d["hs"]
BC_CONF = d["bc"]


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


class VAPHead(nn.Module):
    def __init__(self, input_dim, n_bins=4, type="discrete"):
        super().__init__()
        self.type = type

        self.output_dim = 1
        if type == "comparative":
            self.projection_head = nn.Linear(input_dim, 1)
        else:
            self.total_bins = 2 * n_bins
            if type == "independent":
                self.projection_head = nn.Sequential(
                    nn.Linear(input_dim, self.total_bins),
                    Rearrange("... (c f) -> ... c f", c=2, f=self.total_bins // 2),
                )
                self.output_dim = (2, n_bins)
            else:
                self.n_classes = 2 ** self.total_bins
                self.projection_head = nn.Linear(input_dim, self.n_classes)

    def __repr__(self):
        s = "VAPHead\n"
        s += f"  type: {self.type}"
        s += f"  output: {self.output_dim}"
        return super().__repr__()

    def forward(self, x):
        return self.projection_head(x)


class ProjectionModel(nn.Module):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf

        # Audio Encoder
        self.encoder = Encoder(conf)

        # VAD Conditioning
        self.vad_condition = VadCondition(
            self.encoder.output_dim,
            vad_history=conf["vad_cond"]["vad_history"],
            vad_history_bins=conf["vad_cond"]["vad_history_bins"],
        )

        # Autoregressive
        self.ar = AR(
            input_dim=self.encoder.output_dim,
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

        # Appropriate VAP-head
        self.vap_type = "discrete"
        if conf["vad_projection"]["regression"]:
            if conf["vad_projection"]["comparative"]:
                self.vap_type = "comparative"
            else:
                self.vap_type = "independent"

        n_bins = len(conf["vad_projection"]["bin_times"])
        self.vap_head = VAPHead(
            input_dim=conf["ar"]["dim"], n_bins=n_bins, type=self.vap_type
        )

    @staticmethod
    def load_config(path=None, args=None, format="dict"):
        if path is None:
            path = repo_root() + "/conv_ssl/config/model.yaml"
        return load_config(path, args=args, format=format)

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

    def encode(self, waveform):
        enc_out = self.encoder(waveform)
        z = enc_out.get("q_idx", enc_out["z"])
        return z

    def forward(self, waveform, va, va_history=None):
        out = {}

        # Encode Audio
        z = self.encode(waveform)

        # Vad conditioning
        vc = self.vad_condition(va, va_history)

        # Add vad-conditioning to audio features
        z = z + vc[:, : z.shape[1]]

        # Autoregressive
        z = self.ar(z)["z"]

        out["logits_vp"] = self.vap_head(z)
        return out


class VPModel(pl.LightningModule):
    def __init__(self, conf) -> None:
        super().__init__()
        self.net = ProjectionModel(conf)  # x, vf, vh -> logits
        self.vap_type = self.net.vap_type

        self.VAP = VAP(
            type=self.vap_type,
            bin_times=conf["vad_projection"]["bin_times"],
            frame_hz=self.net.encoder.frame_hz,
            pre_frames=conf["vad_projection"]["regression_pre_frames"],
            threshold_ratio=conf["vad_projection"]["vad_threshold"],
        )  # logits -> zero-shot probs etc

        # conf
        # self.frame_hz = self.net.encoder.frame_hz
        self.conf = conf

        # Metrics
        self.val_metric = None  # self.init_metric(conf, frame_hz=self.frame_hz)
        self.test_metric = None  # set in test if necessary

        # Training params
        self.learning_rate = conf["optimizer"]["learning_rate"]
        self.betas = conf["optimizer"]["betas"]
        self.alpha = conf["optimizer"]["alpha"]
        self.weight_decay = conf["optimizer"]["weight_decay"]
        self.save_hyperparameters()

    @property
    def frame_hz(self):
        return self.net.encoder.frame_hz

    def init_metric(
        self,
        conf,
        frame_hz,
        threshold_pred_shift=0.5,
        threshold_short_long=0.5,
        threshold_bc_pred=0.1,
        bc_pred_pr_curve=False,
        shift_pred_pr_curve=False,
        long_short_pr_curve=False,
    ):
        metric = TurnTakingMetrics(
            hs_kwargs=HS_CONF,
            bc_kwargs=BC_CONF,
            metric_kwargs=METRIC_CONF,
            threshold_pred_shift=threshold_pred_shift,
            threshold_short_long=threshold_short_long,
            threshold_bc_pred=threshold_bc_pred,
            shift_pred_pr_curve=shift_pred_pr_curve,
            bc_pred_pr_curve=bc_pred_pr_curve,
            long_short_pr_curve=long_short_pr_curve,
            frame_hz=self.frame_hz,
        )
        metric = metric.to(self.device)
        return metric

    @staticmethod
    def load_config(path=None, args=None, format="dict"):
        return ProjectionModel.load_config(path, args, format)

    @property
    def run_name(self):
        # Name the run e.g. hubert_44_41
        name = self.conf["encoder"]["type"].replace("_base", "")
        name += f"_{self.conf['ar']['num_layers']}{self.conf['ar']['num_heads']}"
        if self.conf["vad_projection"]["regression"]:
            if self.conf["vad_projection"]["comparative"]:
                name += "_comp"
            else:
                n_bins = len(self.conf["vad_projection"]["bin_times"])
                name += f"_ind_{n_bins}"
        return name

    def summary(self):
        s = "VPModel\n"
        s += f"{self.net}"
        s += f"{self.VAP}"
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

    def calc_losses(self, logits, va_labels, reduction="mean"):
        if self.vap_type == "comparative":
            loss = F.binary_cross_entropy_with_logits(logits, va_labels.unsqueeze(-1))
        elif self.vap_type == "independent":
            loss = F.binary_cross_entropy_with_logits(
                logits, va_labels, reduction=reduction
            )
        else:
            loss = self.net.loss_vad_projection(
                logits, labels=va_labels, reduction=reduction
            )
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
        # Get labels
        va_labels = self.VAP.extract_label(va=batch["vad"])

        # Only keep the relevant vad information
        n_valid = va_labels.shape[1]
        batch["vad"] = batch["vad"][:, :n_valid]
        if "vad_history" in batch:
            batch["vad_history"] = batch["vad_history"][:, :n_valid]

        # Forward pass
        out = self(
            waveform=batch["waveform"],
            va=batch["vad"],
            va_history=batch.get("vad_history", None),
        )
        out["va_labels"] = va_labels

        # Calculate Loss
        loss = self.calc_losses(
            logits=out["logits_vp"],
            va_labels=va_labels,
            reduction=reduction,
        )

        loss = {"vp": loss.mean(), "total": loss.mean()}
        return loss, out, batch

    @torch.no_grad()
    def output(self, batch, out_device="cpu"):
        loss, out, batch = self.shared_step(to_device(batch, self.device))
        probs = self.VAP(logits=out["logits_vp"], va=batch["vad"])
        batch = to_device(batch, out_device)
        out = to_device(out, out_device)
        loss = to_device(loss, out_device)
        probs = to_device(probs, out_device)
        return loss, out, probs, batch

    def training_step(self, batch, batch_idx, **kwargs):
        loss, _, _ = self.shared_step(batch)
        batch_size = batch["vad"].shape[0]

        self.log("loss", loss["total"], batch_size=batch_size)
        self.log("loss_vp", loss["vp"], batch_size=batch_size)

        if "loss_ar" in loss:
            self.log("loss_ar", loss["ar"], batch_size=batch_size)

        return {"loss": loss["total"]}

    def validation_step(self, batch, batch_idx, **kwargs):
        """validation step"""
        if self.val_metric is None:
            self.val_metric = self.init_metric(self.conf, self.frame_hz)
            self.val_metric.to(self.device)

        # extract events for metrics (use full vad including horizon)
        events = self.val_metric.extract_events(va=batch["vad"], max_frame=1000)

        # Regular forward pass
        loss, out, batch = self.shared_step(batch)
        batch_size = batch["vad"].shape[0]

        # log scores
        self.log("val_loss", loss["total"], batch_size=batch_size)

        # Extract other metrics
        turn_taking_probs = self.VAP(logits=out["logits_vp"], va=batch["vad"])
        self.val_metric.update(
            p=turn_taking_probs["p"],
            bc_pred_probs=turn_taking_probs.get("bc_prediction", None),
            events=events,
        )

    def on_test_epoch_start(self):
        if self.test_metric is not None:
            self.test_metric.reset()

    def on_validation_epoch_start(self):
        if self.val_metric is not None:
            self.val_metric.reset()

    def validation_epoch_end(self, outputs) -> None:
        r = self.val_metric.compute()
        for metric_name, values in r.items():
            if metric_name.startswith("pr_curve"):
                continue
            if isinstance(values, dict):
                for val_name, val in values.items():
                    if val_name in ["tp", "tn", "fp", "fn"]:
                        continue
                    self.log(f"val_{metric_name}_{val_name}", val)
            else:
                self.log(f"val_{metric_name}", values)

    def test_step(self, batch, batch_idx, **kwargs):
        if self.test_metric is None:
            self.test_metric = self.init_metric(self.conf, self.frame_hz)
            self.test_metric.to(self.device)

        # extract events for metrics (use full vad including horizon)
        events = self.test_metric.extract_events(va=batch["vad"], max_frame=1000)

        # Regular forward pass
        loss, out, batch = self.shared_step(batch)
        batch_size = batch["vad"].shape[0]

        # log scores
        self.log("test_loss", loss["total"], batch_size=batch_size)

        # Extract other metrics
        turn_taking_probs = self.VAP(logits=out["logits_vp"], va=batch["vad"])
        self.test_metric.update(
            p=turn_taking_probs["p"],
            bc_pred_probs=turn_taking_probs.get("bc_prediction", None),
            events=events,
        )

    def test_epoch_end(self, outputs) -> None:
        r = self.test_metric.compute()
        try:
            for metric_name, values in r.items():
                if metric_name.startswith("pr_curve"):
                    continue
                if isinstance(values, dict):
                    for val_name, val in values.items():
                        if val_name in ["tp", "tn", "fp", "fn"]:
                            continue
                        self.log(f"test/{metric_name}_{val_name}", val)
                else:
                    self.log(f"test/{metric_name}", values)
        except Exception as e:
            print("TEST_EPOCH_END FAILED.", e)

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
    from datasets_turntaking import DialogAudioDM
    from conv_ssl.evaluation.utils import get_checkpoint
    from conv_ssl.utils import to_device

    data_conf = DialogAudioDM.load_config()
    DialogAudioDM.print_dm(data_conf)

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
        vad_hz=100,
        vad_horizon=2,
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=4,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup()

    ################################################
    # Test new
    ################################################
    conf = VPModel.load_config()
    model = VPModel(conf)
    for batch in dm.train_dataloader():
        loss, out, batch = model.shared_step(to_device(batch, model.device))
        print("Loss: ", loss["total"])
        break

    ################################################
    # Load old
    ################################################
    run_path = "120k8fdv"
    checkpoint_path = get_checkpoint(run_path=run_path)
    checkpoint_path = load_paper_versions(checkpoint_path)
    model = VPModel.load_from_checkpoint(checkpoint_path, strict=False)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model = model.eval()
    for batch in dm.train_dataloader():
        loss, out, batch = model.shared_step(to_device(batch, model.device))
        print("Loss: ", loss["total"])
