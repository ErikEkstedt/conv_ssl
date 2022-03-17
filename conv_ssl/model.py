import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pytorch_lightning as pl

from conv_ssl.models import Encoder, AR
from conv_ssl.utils import OmegaConfArgs, repo_root, load_config

from vad_turn_taking.metrics import TurnTakingMetrics
from vad_turn_taking.vad_projection import (
    VadLabel,
    ProjectionCodebook,
    ProjectionIndependent,
)
from vad_turn_taking.vad import VAD
from vad_turn_taking.backchannel import extract_backchannel_prediction_probs_independent


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


class CAPEncoder(nn.Module):
    def __init__(self, input_shape=(2, 40), latent_dim=32, type="conv"):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        self.output_norm = nn.Identity()
        if type == "linear":
            self.net = nn.Sequential(
                nn.Linear(input_shape, latent_dim * 2),
                nn.Tanh(),
                nn.Linear(latent_dim * 2, latent_dim),
            )
        elif type == "conv":
            h = 16
            self.net = nn.Sequential(
                nn.Conv1d(
                    input_shape[0],
                    out_channels=h,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Conv1d(
                    h,
                    out_channels=h,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Flatten(1),
                nn.Linear(h * input_shape[-1], latent_dim),
            )

        self.init_prototypes()

    def extract_negatives(
        self, y, n_negatives=2, within=True, min_shift=95, max_shift=125
    ):
        """
        Arguments:
        y:              torch.tensor, (B, t, 2, 40)
        n_negatives:    int,
        within:         bool, within or between

        Return:
        y_neg:      (n_neg, B, t, 2, 40)
        """

        assert within, f"Only within sampling is defined! but within={within}"

        nb, nt, _, _ = y.shape
        negs = []
        t = torch.arange(nt)
        if within:
            for b in range(nb):
                bs = torch.ones(n_negatives * len(t), dtype=torch.long) * b
                # shift the sequence to avoid same samples, or very similar.
                shift = torch.randint(min_shift, max_shift, (1,))
                # tt = t.roll(shift, 0).repeat(n_negatives)
                tt = t.roll(shift.item(), 0).repeat(n_negatives)
                negs.append(
                    einops.rearrange(y[bs, tt], "(a n) c d -> a n c d", a=n_negatives)
                )
            negs = torch.stack(negs, dim=1)
        return negs

    def _prototype_vector(
        self,
        active_channel,
        silence_prefix=0,
        silence_suffix=0,
        other_active_prefix=0,
        other_active_suffix=0,
    ):
        v = torch.zeros(self.input_shape, dtype=torch.float)

        if silence_prefix > 0:
            v[active_channel, silence_prefix:] = 1.0

        if silence_suffix > 0:
            v[active_channel, -silence_suffix:] = 0.0

        other_channel = 0 if active_channel == 1 else 1

        if other_active_prefix > 0:
            v[other_channel, :other_active_prefix] = 1.0

        if other_active_suffix > 0:
            v[other_channel, -other_active_suffix:] = 1.0

        return v

    def init_prototypes(self):
        n = self.input_shape[-1]
        silence_prefix = n // 4
        bc_other_prefix = silence_prefix // 2

        prototypes = []
        # next speaker on silence
        prototypes.append(self._prototype_vector(0, silence_prefix))
        prototypes.append(self._prototype_vector(1, silence_prefix))
        # next speaker on active
        prototypes.append(
            self._prototype_vector(
                0, silence_prefix, other_active_prefix=silence_prefix
            )
        )
        prototypes.append(
            self._prototype_vector(
                1, silence_prefix, other_active_prefix=silence_prefix
            )
        )
        # BC prediction Silence
        prototypes.append(
            self._prototype_vector(
                0,
                silence_prefix,
                silence_suffix=silence_prefix,
                other_active_suffix=silence_prefix,
            )
        )
        prototypes.append(
            self._prototype_vector(
                1,
                silence_prefix,
                silence_suffix=silence_prefix,
                other_active_suffix=silence_prefix,
            )
        )
        # BC prediction Active
        prototypes.append(
            self._prototype_vector(
                0,
                silence_prefix,
                silence_suffix=silence_prefix,
                other_active_prefix=bc_other_prefix,
                other_active_suffix=silence_prefix,
            )
        )
        prototypes.append(
            self._prototype_vector(
                1,
                silence_prefix,
                silence_suffix=silence_prefix,
                other_active_prefix=bc_other_prefix,
                other_active_suffix=silence_prefix,
            )
        )

        self.prototypes = torch.stack(prototypes)
        self.prototypes_names = {
            "a_next_sil": 1,
            "b_next_sil": 2,
            "a_next_act": 3,
            "b_next_act": 4,
            "a_bc_pred": 5,
            "b_bc_pred": 6,
            "a_bc_pred_act": 7,
            "b_bc_pred_act": 8,
        }
        self.prototypes_idx = {v: k for k, v in self.prototypes_names.items()}

    def get_probs(self, z_pred, vad=None):
        """
        Given the predicted vector at each time step, compare 'prototype' vectors, and match the closest.
        """

        B, T, D = z_pred.shape
        zp = self(self.prototypes.to(z_pred.device))
        zp = zp.unsqueeze(1)  # batch dim

        zpredflat = einops.rearrange(z_pred, "b t d -> (b t) d")
        score = F.cosine_similarity(zpredflat, zp, dim=-1)
        score = einops.rearrange(score, "n (b t) -> n b t", b=B)  # (N, B, T)

        # SILENCE probs
        # ON SILENCE: compare prototype score for A is next speaker with B
        # compare the normalized distance and treat as a probability
        # shift and scale scores [-1, 1] -> [0, 2] -> [0, 1]
        # i.e. A=0.3 and B=-.2 -> 1.3, 0.2 -> 0.65, 0.1 ->
        a_sil = (score[0] + 1) / 2
        b_sil = (score[1] + 1) / 2
        sum = a_sil + b_sil
        sil_probs = torch.stack((a_sil, b_sil), dim=-1) / sum.unsqueeze(-1)

        # ACTIVE
        # Compare a-active with B-is-next from above -> renormalize etc
        a_act = (score[2] + 1) / 2
        a_act = a_act / (a_act + b_sil)
        b_act = (score[3] + 1) / 2
        b_act = b_act / (b_act + a_sil)
        act_probs = torch.stack((a_act, b_act), dim=-1)

        # BC prediction SILENCE
        # We only use the the normalized score for the backchannel
        # prediction -> using a threshold during test to find where the best value is
        a_bc_pred_sil = (score[4] + 1) / 2
        b_bc_pred_sil = (score[5] + 1) / 2

        # BC prediction ACTIVE
        # We only use the the normalized score for the backchannel
        # prediction -> using a threshold during test to find where the best value is
        a_bc_pred_act = (score[6] + 1) / 2
        b_bc_pred_act = (score[7] + 1) / 2

        ################################################################
        # Final "Probabilities"
        ################################################################
        # Extract appropriate values on certain states of the input
        p_a = torch.zeros_like(sil_probs[..., 0])
        p_b = torch.zeros_like(sil_probs[..., 0])
        p_a_bc_pred = torch.zeros_like(sil_probs[..., 0])
        p_b_bc_pred = torch.zeros_like(sil_probs[..., 0])

        # dialog states
        ds = VAD.vad_to_dialog_vad_states(vad)
        silence = ds == 1
        a_current = ds == 0
        b_current = ds == 3
        both = ds == 2

        # silence
        w = torch.where(silence)
        p_a[w] = sil_probs[w][..., 0]
        p_b[w] = sil_probs[w][..., 1]
        p_a_bc_pred[w] = a_bc_pred_sil[w]
        p_b_bc_pred[w] = b_bc_pred_sil[w]

        # A current speaker
        # Given only A is speaking we use the 'active' probability of B being the next speaker
        w = torch.where(a_current)
        p_a[w] = 1 - act_probs[w][..., 1]  # P_a = 1-P_b
        p_b[w] = act_probs[w][..., 1]  # P_b
        p_a_bc_pred[w] = 1 - b_bc_pred_act[w]
        p_b_bc_pred[w] = b_bc_pred_act[w]

        # B current speaker
        w = torch.where(b_current)
        p_a[w] = act_probs[w][..., 0]  # P_a for A being next speaker, while B is active
        p_b[w] = 1 - act_probs[w][..., 0]  # P_b = 1-P_a
        p_a_bc_pred[w] = a_bc_pred_act[w]
        p_b_bc_pred[w] = 1 - a_bc_pred_act[w]

        # Both
        # P_a_prior=A is next (active)
        # P_b_prior=B is next (active)
        # We the compare/renormalize given the two values of A/B is the next speaker
        # sum = P_a_prior+P_b_prior
        # P_a = P_a_prior / sum
        # P_b = P_b_prior / sum
        w = torch.where(both)
        # Re-Normalize and compare next-active
        sum = act_probs[w][..., 0] + act_probs[w][..., 1]
        p_a[w] = act_probs[w][..., 0] / sum
        p_b[w] = act_probs[w][..., 1] / sum
        p = torch.stack((p_a, p_b), dim=-1)
        bc_probs = torch.stack((p_a_bc_pred, p_b_bc_pred), dim=-1)
        return {"p": p, "bc_prediction": bc_probs}

    def nce_loss(self, z_pred, y, n_negatives=5):
        b, t, c, d = y.shape

        # Positives
        yp = einops.rearrange(y, "b t ... -> (b t) ...")
        zp = einops.rearrange(self(yp), "(b t) ... -> b t ...", b=b, t=t)
        zp = zp.unsqueeze(0)  # add neg/pos dimension

        # Negatives
        y_neg = self.extract_negatives(y, n_negatives=n_negatives)
        yn = einops.rearrange(y_neg, "n b t ... -> (n b t) ...")
        zn = einops.rearrange(
            self(yn), "(n b t) ... -> n b t ... ", n=n_negatives, b=b, t=t
        )

        # Join positives/negatives
        zjoint = torch.cat((zp, zn))

        # calculate similarity
        score = F.cosine_similarity(z_pred, zjoint, dim=-1)
        score = einops.rearrange(score, "n b t -> (b t) n")
        label = torch.zeros_like(score[:, 0], dtype=torch.long)
        loss = F.cross_entropy(score, label)
        return loss

    def forward(self, x):
        z = self.net(x)
        z = self.output_norm(z)
        return z


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
            if conf["vad_projection"]["comparative"]:
                self.projection_head = nn.Linear(input_dim, 1)
            else:
                cf = len(conf["vad_projection"]["bin_times"]) * 2
                self.projection_head = nn.Sequential(
                    nn.Linear(input_dim, cf),
                    Rearrange("... (c f) -> ... c f", c=2, f=cf // 2),
                )
        elif conf["vad_projection"]["latent"]:
            self.projection_head = nn.Linear(
                input_dim, conf["vad_projection"]["latent_dim"]
            )
            bins = len(conf["vad_projection"]["bin_times"])
            self.future_encoder = CAPEncoder(
                input_shape=(2, bins),
                latent_dim=conf["vad_projection"]["latent_dim"],
                type="conv",
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


class VadProjectionTask(pl.LightningModule):
    def next_speaker_probs_independent(self, logits, vad=None):
        """Probabilities for turn-taking task given the INDEPENDENT model"""

        def _normalize_reg_probs(probs):
            probs = probs.sum(dim=-1)
            tot = probs.sum(dim=-1, keepdim=True)
            # renormalize for comparison
            probs = probs / tot
            return probs

        # Compare chosen subset of next-speaker-activity
        probs = logits.sigmoid()
        next_probs = _normalize_reg_probs(probs)
        pre_probs = _normalize_reg_probs(probs[..., :, self.pre_frames :])
        bc_prediction = extract_backchannel_prediction_probs_independent(probs)
        return {"p": next_probs, "pre_probs": pre_probs, "bc_prediction": bc_prediction}

    def next_speaker_probs_comparative(self, logits):
        """
        Probabilities for turn-taking task given the COMPARATIVE model
        """
        probs = logits.sigmoid()
        next_probs = torch.cat((probs, 1 - probs), dim=-1)
        return {"p": next_probs, "pre_probs": next_probs}

    def get_next_speaker_probs(self, logits, vad=None):
        if self.regression:
            if self.conf["vad_projection"]["comparative"]:
                out = self.next_speaker_probs_comparative(logits)
            else:
                # out = self.next_speaker_probs_independent(logits, vad)
                out = self.projection_codebook.get_probs(logits, vad=vad)
        elif self.conf["vad_projection"]["latent"]:
            out = self.net.future_encoder.get_probs(z_pred=logits, vad=vad)
        else:
            out = self.projection_codebook.get_probs(logits, vad)
        return out


# class VPModel(pl.LightningModule):
class VPModel(VadProjectionTask):
    @staticmethod
    def load_config(path=None, args=None, format="dict"):
        return ProjectionModel.load_config(path, args, format)

    def __init__(self, conf) -> None:
        super().__init__()

        # tmp fix
        if "latent" not in conf["vad_projection"]:
            conf["vad_projection"]["latent"] = False
            conf["vad_projection"]["latent_dim"] = -1
        self.loss_vector = torch.zeros(1000, dtype=torch.float)
        self.loss_n = torch.tensor([0])

        self.net = ProjectionModel(conf)

        # Extract labels
        self.vad_label_maker = VadLabel(
            bin_times=conf["vad_projection"]["bin_times"],
            vad_hz=self.net.encoder.frame_hz,
            threshold_ratio=conf["vad_projection"]["vad_threshold"],
        )

        # conf
        self.frame_hz = self.net.encoder.frame_hz
        self.conf = conf

        # Metrics
        self.val_metric = None  # self.init_metric(conf, frame_hz=self.frame_hz)
        self.test_metric = None  # set in test if necessary

        # NOTE: Unused?
        self.event_dict = dict(
            bin_times=conf["vad_projection"]["bin_times"],
            vad_threshold=conf["vad_projection"]["vad_threshold"],
            pred_threshold=conf["vad_projection"]["pred_threshold"],
            event_min_context=conf["vad_projection"]["event_min_context"],
            event_min_duration=conf["vad_projection"]["event_min_duration"],
            event_horizon=conf["vad_projection"]["event_horizon"],
            event_start_pad=conf["vad_projection"]["event_start_pad"],
            event_target_duration=conf["vad_projection"]["event_target_duration"],
            event_bc_pre_silence=conf["vad_projection"]["event_bc_pre_silence"],
            event_bc_post_silence=conf["vad_projection"]["event_bc_post_silence"],
            event_bc_max_active=conf["vad_projection"]["event_bc_max_active"],
            frame_hz=self.net.encoder.frame_hz,
        )

        # only use discrete codes if necessary
        self.regression = conf["vad_projection"]["regression"]
        self.pre_frames = conf["vad_projection"]["regression_pre_frames"]

        if self.regression:
            self.projection_codebook = ProjectionIndependent(
                pre_frames=conf["vad_projection"]["regression_pre_frames"],
                bin_times=conf["vad_projection"]["bin_times"],
                frame_hz=self.net.encoder.frame_hz,
            )
        if not (self.regression or self.conf["vad_projection"]["latent"]):
            self.projection_codebook = ProjectionCodebook(
                bin_times=conf["vad_projection"]["bin_times"],
                frame_hz=self.net.encoder.frame_hz,
            )

        # Training params
        self.learning_rate = conf["optimizer"]["learning_rate"]
        self.betas = conf["optimizer"]["betas"]
        self.alpha = conf["optimizer"]["alpha"]
        self.weight_decay = conf["optimizer"]["weight_decay"]
        self.save_hyperparameters()

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
        **event_kwargs,
    ):
        # metric = TurnTakingMetrics(
        #     # don't extract events before a certain context is known
        #     min_context=conf["vad_projection"]["event_min_context"],
        #     # the event-horizon (same as vad)
        #     horizon=conf["vad_projection"]["event_horizon"],
        #     # time after activity to start use frames for prediction
        #     start_pad=conf["vad_projection"]["event_start_pad"],
        #     # number of frames to use in metrics/event
        #     target_duration=conf["vad_projection"]["event_target_duration"],
        #     # the frame-hz of vad-representation
        #     frame_hz=frame_hz,
        #     # time at eot to predict shift/holf
        #     pre_active=conf["vad_projection"]["event_pre"],
        #     # silence before activity to be considered BC
        #     bc_pre_silence=conf["vad_projection"]["event_bc_pre_silence"],
        #     # silence after activity to be considered BC
        #     bc_post_silence=conf["vad_projection"]["event_bc_post_silence"],
        #     # longest activity to be considered BC
        #     bc_max_active=conf["vad_projection"]["event_bc_max_active"],
        #     # The amount of time prior a backchannel to infer bc-prediciton stats
        #     bc_prediction_window=conf["vad_projection"]["event_bc_prediction_window"],
        #     threshold=0.5,  # f1 threshold
        #     threshold_bc_ongoing=conf["vad_projection"]["event_bc_ongoing_threshold"],
        #     threshold_bc_pred=conf["vad_projection"]["event_bc_pred_threshold"],
        #     bc_pred_pr_curve=bc_pred_pr_curve,
        #     discrete=not conf["vad_projection"]["regression"],  # discrete model or not
        # )
        metric = TurnTakingMetrics(
            threshold_pred_shift=threshold_pred_shift,
            threshold_short_long=threshold_short_long,
            threshold_bc_pred=threshold_bc_pred,
            bc_pred_pr_curve=bc_pred_pr_curve,
            shift_pred_pr_curve=shift_pred_pr_curve,
            long_short_pr_curve=long_short_pr_curve,
            frame_hz=frame_hz,
            **event_kwargs,
        )
        metric = metric.to(self.device)
        return metric

    @property
    def run_name(self):
        # Name the run e.g. hubert_44_41
        name = self.conf["encoder"]["type"].replace("_base", "")
        name += f"_{self.conf['ulm']['num_layers']}{self.conf['ulm']['num_heads']}"
        name += f"_{self.conf['ar']['num_layers']}{self.conf['ar']['num_heads']}"
        if self.conf["vad_projection"]["regression"]:
            if self.conf["vad_projection"]["comparative"]:
                name += "_comp"
            else:
                n_bins = len(self.conf["vad_projection"]["bin_times"])
                name += f"_ind_{n_bins}"
        elif self.conf["vad_projection"]["latent"]:
            name += "_latent"
        return name

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

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

    def calc_losses(self, out, vad_projection_window, input_ids=None, reduction="mean"):
        loss = {}

        # Vad Projection Loss
        if self.conf["vad_projection"]["regression"]:
            if self.conf["vad_projection"]["comparative"]:
                loss["vp"] = F.binary_cross_entropy_with_logits(
                    out["logits_vp"], vad_projection_window.unsqueeze(-1)
                )
            else:
                if self.conf["vad_projection"]["regression_loss"] == "mse":
                    loss["vp"] = F.mse_loss(
                        out["logits_vp"].sigmoid(), vad_projection_window
                    )
                elif self.conf["vad_projection"]["regression_loss"] in ["mae", "l1"]:
                    loss["vp"] = F.l1_loss(
                        out["logits_vp"].sigmoid(), vad_projection_window
                    )
                else:  # BCE
                    loss["vp"] = F.binary_cross_entropy_with_logits(
                        out["logits_vp"], vad_projection_window
                    )
        elif self.conf["vad_projection"]["latent"]:
            z_pred = out["logits_vp"]  # Not logits but projected z vectors
            loss["vp"] = self.net.future_encoder.nce_loss(
                z_pred, vad_projection_window, n_negatives=5
            )
        else:
            # Vad Projection Loss
            vad_labels = self.projection_codebook(vad_projection_window)
            loss["vp"] = self.net.loss_vad_projection(
                logits=out["logits_vp"], labels=vad_labels, reduction=reduction
            )

        # ULM Loss
        if self.net.ulm is not None and input_ids is not None:
            if self.conf["vad_projection"]["latent"]:
                raise NotImplementedError(
                    "Can't use 'latent' contrastive training with ULM yet..."
                )
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
        if (
            "comparative" in self.conf["vad_projection"]
            and self.conf["vad_projection"]["comparative"]
        ):  # scalar value
            vad_projection_window = self.vad_label_maker.comparative_activity(
                batch["vad"]
            )
        else:  # onehot window projection
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
            # reduction=reduction,
            reduction="none",
        )

        self.loss_vector += loss["vp"].sum(0).cpu()
        self.loss_n += batch["vad"].shape[0]

        loss = {"vp": loss["vp"].mean(), "total": loss["total"].mean()}
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
        """validation step"""
        if self.val_metric is None:
            self.val_metric = self.init_metric(self.conf, self.frame_hz)
            self.val_metric.to(self.device)

        # extract events for metrics (use full vad including horizon)
        events = self.val_metric.extract_events(batch["vad"])

        # Regular forward pass
        loss, out, batch = self.shared_step(batch)
        batch_size = batch["vad"].shape[0]

        # log scores
        self.log("val_loss", loss["total"], batch_size=batch_size)
        self.log("val_loss_vp", loss["vp"], batch_size=batch_size)
        if "loss_ar" in loss:
            self.log("val_loss_ar", loss["ar"], batch_size=batch_size)

        # Extract other metrics
        turn_taking_probs = self.get_next_speaker_probs(
            out["logits_vp"], vad=batch["vad"]
        )
        self.val_metric.update(
            p=turn_taking_probs["p"],
            pw=turn_taking_probs.get("pw", None),  # only in discrete
            pre_probs=turn_taking_probs.get("pre_probs", None),  # only in independent
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
        events = self.test_metric.extract_events(batch["vad"])

        # Regular forward pass
        loss, out, batch = self.shared_step(batch)
        batch_size = batch["vad"].shape[0]

        # log scores
        self.log("test_loss", loss["total"], batch_size=batch_size)
        self.log("test_loss_vp", loss["vp"], batch_size=batch_size)
        if "loss_ar" in loss:
            self.log("test_loss_ar", loss["ar"], batch_size=batch_size)

        # Extract other metrics
        turn_taking_probs = self.get_next_speaker_probs(
            out["logits_vp"], vad=batch["vad"]
        )
        self.test_metric.update(
            p=turn_taking_probs["p"],
            pw=turn_taking_probs.get("pw", None),  # only in discrete
            pre_probs=turn_taking_probs.get("pre_probs", None),  # only in independent
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


def debug_capencoder():
    from datasets_turntaking import DialogAudioDM

    input_shape = (2, 40)
    N = 1000
    B = 4
    D = 32
    y = torch.randint(0, 2, (B, N, *input_shape), dtype=torch.float)
    print("y: ", tuple(y.shape))

    #######################################################################
    # TEST CAPENCODER
    enc = CAPEncoder(input_shape=input_shape, latent_dim=D, type="conv")
    # self = enc

    # z prediction
    z_pred = torch.rand((B, N, D))
    print("z_pred: ", tuple(z_pred.shape))
    # contrastive loss for conv
    l = enc.nce_loss(z_pred, y, n_negatives=5)
    print("loss: ", l)
    #######################################################################

    out = enc.get_probs(z_pred, vad)

    data_conf = DialogAudioDM.load_config()
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
        vad_horizon=2.0,
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=4,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup()
    print(dm)
    diter = iter(dm.val_dataloader())
    batch = next(diter)

    conf = VPModel.load_config(path="conv_ssl/config/model_latent.yaml")
    model = VPModel(conf)

    loss, out, batch = model.shared_step(batch)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from datasets_turntaking import DialogAudioDM
    from conv_ssl.utils import count_parameters

    parser = ArgumentParser()
    parser = DialogAudioDM.add_data_specific_args(parser)
    args = parser.parse_args()
    data_conf = DialogAudioDM.load_config(path=args.data_conf, args=args)
    DialogAudioDM.print_dm(data_conf, args)

    # Model
    conf = VPModel.load_config(path="conv_ssl/config/model_independent.yaml")
    # conf["vad_projection"]["regression"] = True
    # conf["vad_projection"]["regression"] = True
    # conf["vad_projection"]["comparative"] = True
    # conf["vad_projection"]["bin_times"] = [0.05] * 40
    # conf["vad_projection"][
    #     "regression_loss"
    # ] = "bce"  # 'mae', 'l1', 'mse' otherwise 'bce'
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
        vad_horizon=sum(model.conf["vad_projection"]["bin_times"]),
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=4,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup()
    print(dm)
    diter = iter(dm.val_dataloader())

    batch = next(diter)
    if cuda:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to("cuda")
    loss, out, batch = model.shared_step(batch)
    turn_taking_probs = model.get_next_speaker_probs(out["logits_vp"], vad=batch["vad"])
    for k, v in turn_taking_probs.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    probs = out["logits_vp"].sigmoid()
