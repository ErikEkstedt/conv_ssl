import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pytorch_lightning as pl

from conv_ssl.models import Encoder, AR
from conv_ssl.utils import repo_root, load_config, to_device
from vap_turn_taking import VAP, TurnTakingMetrics


class VadCondition(nn.Module):
    def __init__(self, dim, va_history=False, va_history_bins=5) -> None:
        super().__init__()
        self.dim = dim

        # Vad Condition
        # vad: 2 one-hot encodings
        # self.vad_condition = nn.Embedding(num_embeddings=2, embedding_dim=dim)
        self.va_condition = nn.Linear(2, dim)

        if va_history:
            self.va_history = nn.Linear(va_history_bins, dim)

        self.ln = nn.LayerNorm(dim)
        # self.init()

    def init(self):
        # init orthogonal vad vectors
        nn.init.orthogonal_(self.va_condition.weight.data)

    def forward(self, vad, va_history=None):
        v_cond = self.va_condition(vad)

        # Add vad-history information
        if va_history is not None:
            v_cond += self.va_history(va_history)

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
        self.encoder = Encoder(conf["encoder"])

        # VAD Conditioning
        self.vad_condition = VadCondition(
            self.encoder.output_dim,
            va_history=conf["va_cond"]["history"],
            va_history_bins=conf["va_cond"]["history_bins"],
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
        self.vap_type = conf["vap"]["type"]
        n_bins = len(conf["vap"]["bin_times"])
        self.vap_head = VAPHead(
            input_dim=conf["ar"]["dim"], n_bins=n_bins, type=self.vap_type
        )

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
        self.conf = conf

        # Network
        self.net = ProjectionModel(conf["model"])  # x, vf, vh -> logits
        self.vap_type = conf["model"]["vap"]["type"]

        # VAP: labels, logits -> zero-shot probs
        self.VAP = VAP(
            type=conf["model"]["vap"]["type"],
            bin_times=conf["model"]["vap"]["bin_times"],
            frame_hz=self.net.encoder.frame_hz,
            pre_frames=conf["model"]["vap"]["pre_frames"],
            threshold_ratio=conf["model"]["vap"]["bin_threshold"],
        )

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
        conf=None,
        threshold_pred_shift=None,
        threshold_short_long=None,
        threshold_bc_pred=None,
        bc_pred_pr_curve=False,
        shift_pred_pr_curve=False,
        long_short_pr_curve=False,
    ):
        if conf is None:
            conf = self.conf

        if threshold_pred_shift is None:
            threshold_pred_shift = conf["events"]["threshold"]["S_pred"]

        if threshold_bc_pred is None:
            threshold_bc_pred = conf["events"]["threshold"]["BC_pred"]

        if threshold_short_long is None:
            threshold_short_long = conf["events"]["threshold"]["SL"]

        metric = TurnTakingMetrics(
            hs_kwargs=conf["events"]["SH"],
            bc_kwargs=conf["events"]["BC"],
            metric_kwargs=conf["events"]["metric"],
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

    @property
    def run_name(self):
        conf = self.conf["model"]
        name = conf["encoder"]["name"]
        name += f"_{conf['ar']['num_layers']}{conf['ar']['num_heads']}"
        if self.vap_type != "discrete":
            if self.vap_type == "comparative":
                name += "_comp"
            else:
                n_bins = len(self.conf["vap"]["bin_times"])
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
            lr=self.conf["optimizer"]["learning_rate"],
            betas=self.conf["optimizer"]["betas"],
            weight_decay=self.conf["optimizer"]["weight_decay"],
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
        if self.test_metric is None:
            self.test_metric = self.init_metric()
            self.test_metric.to(self.device)

        if self.test_metric is not None:
            self.test_metric.reset()

    def on_validation_epoch_start(self):
        if self.val_metric is None:
            self.val_metric = self.init_metric()
            self.val_metric.to(self.device)

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
