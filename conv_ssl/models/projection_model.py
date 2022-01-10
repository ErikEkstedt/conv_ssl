from os.path import join
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

import einops
from einops.layers.torch import Rearrange

from conv_ssl.models.autoregressive import AR
from conv_ssl.models.projection_head import ActivityProjectionHead
from conv_ssl.utils import repo_root, load_config

DEFAULT_CONF_PATH = join(repo_root(), "conv_ssl/config/ulm.yaml")


class ProjectionModel(nn.Module):
    """
    A Unit Language Model (like GSLM) with the addition of an activity projection head
    """

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.bin_sizes = conf["vad_class_prediction"]["bin_sizes"]
        self.bin_per_speaker = len(self.bin_sizes)
        self.n_vad_classes = conf["vad_class_prediction"]["n_classes"]

        # loss scaling AR vs VP
        self.alpha = conf["optimizer"]["alpha"]

        # Tier 1 Unit AutoRegressive Language Model
        self.tier1_codebook = nn.Embedding(
            num_embeddings=conf["quantizer"]["n_codes"],
            embedding_dim=conf["tier1"]["dim"],
        )

        # Vad Condition
        # vad: 2 one-hot encodings
        # so this layer is essentially an embedding with 2 indicies
        self.vad_condition = nn.Linear(2, conf["tier1"]["dim"])
        d_history = 5 * 2  # 5 history bins, 2 speakers: (B T 5 2)
        self.vad_history = nn.Sequential(
            Rearrange("B T b c -> B T (b c)"),
            nn.Linear(d_history, conf["tier1"]["dim"]),
        )
        self.cond_norm = nn.LayerNorm(conf["tier1"]["dim"])

        self.tier1 = None
        if conf["tier1"]["num_layers"] > 0:
            self.tier1 = AR(
                input_dim=conf["quantizer"]["dim"],
                dim=conf["tier1"]["dim"],
                num_layers=conf["tier1"]["num_layers"],
                dropout=conf["tier1"]["dropout"],
                ar=conf["tier1"]["type"],
                transfomer_kwargs=dict(
                    num_heads=conf["tier1"]["num_heads"],
                    dff_k=conf["tier1"]["dff_k"],
                    use_pos_emb=conf["tier1"]["use_pos_emb"],
                    abspos=conf["tier1"]["abspos"],
                    sizeSeq=conf["tier1"]["sizeSeq"],
                ),
            )
            self.ar_head = nn.Linear(conf["tier1"]["dim"], conf["quantizer"]["n_codes"])

        # Tier 2 Projection Model
        self.tier2 = None
        if conf["tier2"]["num_layers"] > 0:
            self.tier2 = AR(
                input_dim=conf["tier1"]["dim"],
                dim=conf["tier2"]["dim"],
                num_layers=conf["tier2"]["num_layers"],
                dropout=conf["tier2"]["dropout"],
                ar=conf["tier2"]["type"],
                transfomer_kwargs=dict(
                    num_heads=conf["tier2"]["num_heads"],
                    dff_k=conf["tier2"]["dff_k"],
                    use_pos_emb=conf["tier2"]["use_pos_emb"],
                    abspos=conf["tier2"]["abspos"],
                    sizeSeq=conf["tier2"]["sizeSeq"],
                ),
            )
            self.projection_head = ActivityProjectionHead(
                input_size=conf["tier2"]["dim"],
                bin_sizes=[20, 40, 60, 80],
                threshold_ratio=0.5,
                regression=conf["vad_class_prediction"]["regression"],
            )
        else:
            # append vad bits for current step
            self.projection_head = ActivityProjectionHead(
                input_size=conf["tier1"]["dim"],
                bin_sizes=[20, 40, 60, 80],
                threshold_ratio=0.5,
                regression=conf["vad_class_prediction"]["regression"],
            )

    @staticmethod
    def default_config_path():
        return DEFAULT_CONF_PATH

    @staticmethod
    def load_config(path=None, args=None, format="dict") -> Dict:
        if path is None:
            path = ProjectionModel.default_config_path()
        return load_config(path, args=args, format=format)

    def calc_loss_ar(self, logits_ar, input_ids, reduction="mean"):
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

    def calc_losses(self, out, vad_labels, input_ids=None):
        loss = {}

        # Calculate Projection Loss
        loss["vp"] = self.projection_head.loss_function(
            logits=out["logits_vp"], labels=vad_labels
        )

        # Calculate AR (ULM) Loss
        if self.conf["tier1"]["num_layers"] > 0 and input_ids is not None:
            loss["ar"] = self.calc_loss_ar(out["logits_ar"], input_ids)
            loss["total"] = self.alpha * loss["vp"] + (1 - self.alpha) * loss["ar"]
        else:
            loss["total"] = loss["vp"]
        return loss

    def condition_on_vad(self, z, vad, vad_history=None):
        v_cond = self.vad_condition(vad)
        if vad_history is not None:
            v_cond += self.vad_history(vad_history)
        return self.cond_norm(z + v_cond)

    def forward(self, input_ids, vad, vad_history=None):
        out = {}
        z = self.tier1_codebook(input_ids)

        if self.tier1 is not None and self.tier2 is not None:
            # Both Tiers
            # Tier 1
            z = self.tier1(z)["z"]
            out["logits_ar"] = self.ar_head(z)
            # Tier 2
            z = self.condition_on_vad(z, vad, vad_history)
            z = self.tier2(z)["z"]
            out["z"] = z
        elif self.tier1 is not None:
            z = self.condition_on_vad(z, vad, vad_history)
            z = self.tier1(z)["z"]
            out["z"] = z
            out["logits_ar"] = self.ar_head(z)
        else:
            z = self.condition_on_vad(z, vad, vad_history)
            z = self.tier2(z)["z"]
            out["z"] = z

        out["logits_vp"] = self.projection_head(out["z"])
        return out


if __name__ == "__main__":
    from argparse import ArgumentParser
    from datasets_turntaking.dm_dialog_audio import DialogAudioDM, print_dm

    parser = ArgumentParser()
    parser = DialogAudioDM.add_data_specific_args(parser)
    args = parser.parse_args()
    data_conf = DialogAudioDM.load_config(path=args.data_conf, args=args)
    data_conf["dataset"]["vad_history"] = True
    data_conf["dataset"]["type"] = "sliding"
    print_dm(data_conf, args)

    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        audio_include_ratio=data_conf["dataset"]["audio_include_ratio"],
        audio_context_duration=data_conf["dataset"]["audio_context_duration"],
        ipu_min_time=data_conf["dataset"]["ipu_min_time"],
        ipu_pause_time=data_conf["dataset"]["ipu_pause_time"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hop_time=data_conf["dataset"]["vad_hop_time"],
        vad_bin_sizes=data_conf["dataset"]["vad_bin_sizes"],
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=1,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup()

    conf = ProjectionModel.load_config()
    conf["tier1"]["dim"] = 64
    conf["tier2"]["dim"] = conf["tier1"]["dim"]
    conf["tier1"]["num_layers"] = 1
    conf["tier1"]["num_heads"] = 1
    conf["tier2"]["num_layers"] = 1
    conf["tier2"]["num_heads"] = 1
    conf["vad_class_prediction"]["regression"] = False
    model = ProjectionModel(conf)

    diter = iter(dm.val_dataloader())
    batch = next(diter)

    x = torch.randint(0, 100, (1, 1000))
    out = model(x, batch["vad"])
    print("logits_vp: ", tuple(out["logits_vp"].shape))
    l = model.calc_losses(out, vad_labels=batch["vad_label"], input_ids=x)
