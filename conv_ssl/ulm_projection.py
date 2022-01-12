from os.path import join
from typing import Any, Dict
import matplotlib as mpl

import torch
import pytorch_lightning as pl

mpl.use("Agg")

from conv_ssl.models import ProjectionModel, EncoderPretrained, CHECKPOINTS, MODEL_HZ
from conv_ssl.vad_pred_animation import VadPredAnimator
from conv_ssl.utils import OmegaConfArgs, repo_root, load_config

DEFAULT_CONFIG = join(repo_root(), "conv_ssl/config/ulm_wavlm.yaml")


class ULMProjection(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        # Encoder
        self.encoder = self._build_encoder()

        # ULMProjection
        self.ulm_projection = ProjectionModel(conf)

        # Training params
        self.learning_rate = conf["optimizer"]["learning_rate"]
        self.betas = conf["optimizer"]["betas"]
        # Temporary for older checkpoints
        if "alpha" in conf["optimizer"]:
            self.alpha = conf["optimizer"]["alpha"]
        self.weight_decay = conf["optimizer"]["weight_decay"]
        self.save_hyperparameters()

    def _build_encoder(self, conf=None, load=False):
        if conf is None:
            conf = self.conf
        if conf["encoder"]["pretrained"]:
            return EncoderPretrained(conf, load=load)
        else:
            raise NotImplementedError("implement end-to-end training")

    @property
    def vad_projection_codebook(self):
        return self.ulm_projection.projection_head.vad_projection_codebook

    @property
    def projection_head(self):
        return self.ulm_projection.projection_head

    @property
    def run_name(self):
        # Name the run e.g. hubert_44_41
        name = self.conf["encoder"]["type"].replace("_base", "")
        name += f"_{self.conf['tier1']['num_layers']}{self.conf['tier1']['num_heads']}"
        name += f"_{self.conf['tier2']['num_layers']}{self.conf['tier2']['num_heads']}"
        if self.conf["vad_class_prediction"]["regression"]:
            name += "_reg"
        return name

    @staticmethod
    def default_config_path():
        return DEFAULT_CONFIG

    @staticmethod
    def load_config(path=None, args=None, format="dict") -> Dict:
        if path is None:
            path = ULMProjection.default_config_path()
        return load_config(path, args=args, format=format)

    def forward(self, waveform=None, input_ids=None, vad=None, vad_history=None):
        assert (
            waveform is not None or input_ids is not None
        ), "must provide either `waveform` or `input_ids`"

        out = {}
        if input_ids is None:
            input_ids = self.encoder.get_embeddings(waveform)
            out["input_ids"] = input_ids

        # logits_ar & logits_vp
        o = self.ulm_projection(input_ids=input_ids, vad=vad, vad_history=vad_history)
        out.update(o)
        return out

    def calc_losses(self, out, vad_labels, input_ids, reduction="mean"):
        return self.ulm_projection.calc_losses(
            out, vad_labels, input_ids, reduction=reduction
        )

    def animate_sample(
        self,
        input_ids=None,
        waveform=None,
        vad=None,
        frame_step=5,  # 50 hz
        path="/tmp/ulm_projection_vid.mp4",
    ):
        assert vad.ndim == 2, "input_ids must be (N, 2)"

        if input_ids is not None:
            assert input_ids.ndim == 1, "input_ids must be (N, )"
            out = self(
                input_ids=input_ids.unsqueeze(0).to(self.device),
                vad=vad.unsqueeze(0).to(self.device),  # add batch and move to device
            )
        else:
            assert waveform.ndim == 1, "waveform must be (N_s, )"
            out = self(
                waveform=waveform.unsqueeze(0).to(self.device),
                vad=vad.unsqueeze(0).to(self.device),  # add batch and move to device
            )

        # Greedy vad prediction
        vad_pred = out["logits_vp"][0].argmax(dim=-1)  # omit batch
        vad_pred_oh = self.vad_projection_codebook(vad_pred)
        steps = vad_pred.shape[0]
        ani = VadPredAnimator(
            waveform=waveform,
            vad=vad,
            vad_label_oh=vad_pred_oh.view(
                steps, 2, self.ulm_projection.bin_per_speaker
            ),
            bin_sizes=self.ulm_projection.bin_sizes,
            frame_step=frame_step,
        )
        ani.save_animation(path)
        return path

    def fix_batch_hz(self, batch):
        return self.encoder.fix_batch_hz(batch)

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
        if "q" in batch:  # check if precomputed indices are available
            vad_history = None
            if "vad_history" in batch:
                vad_history = batch["vad_history"]
            out = self(input_ids=batch["q"], vad=batch["vad"], vad_history=vad_history)
            input_ids = batch["q"]
        else:
            # If dataset have units it has also changed the VAD so only do this here
            batch = self.fix_batch_hz(batch)
            vad_history = None
            if "vad_history" in batch:
                vad_history = batch["vad_history"]
            out = self(
                waveform=batch["waveform"], vad=batch["vad"], vad_history=vad_history
            )
            input_ids = out["input_ids"]

        batch_size = input_ids.shape[0]
        loss = self.calc_losses(
            out, input_ids=input_ids, vad_labels=batch["vad_label"], reduction=reduction
        )
        return loss, out, batch, batch_size

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        If trained without and encoder (e.g. hubert) we still
        want to keep the state dict and kmeans vector of the encoder used for
        the training features.
        """
        # save encoder state_dict if not in use
        if self.encoder is None:
            # Load hubert encodings and kmeans to include in checkpoint
            if self.conf["encoder"]["pretrained"]:
                encoder_state_dict = torch.load(
                    CHECKPOINTS[self.conf["encoder"]["type"]]
                )
                for name, weight in encoder_state_dict.items():
                    checkpoint["state_dict"]["encoder.encoder." + name] = weight
            checkpoint["state_dict"]["encoder.quantizer.emb.weight"] = torch.load(
                self.conf["quantizer"]["vector_path"], map_location="cpu"
            )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

    def training_step(self, batch, batch_idx, **kwargs):
        loss, out, _, batch_size = self.shared_step(batch)
        self.log("loss", loss["total"], batch_size=batch_size)
        self.log("loss_vp", loss["vp"], batch_size=batch_size)
        if self.conf["tier1"]["num_layers"] > 0:
            self.log("loss_ar", loss["ar"], batch_size=batch_size)
        return loss["total"]

    def validation_step(self, batch, batch_idx, **kwargs):
        loss, out, batch, batch_size = self.shared_step(batch)
        self.log("val_loss", loss["total"], batch_size=batch_size)
        self.log("val_loss_vp", loss["vp"], batch_size=batch_size)
        if self.conf["tier1"]["num_layers"] > 0:
            self.log("loss_ar", loss["ar"], batch_size=batch_size)

        m = self.projection_head.prepare_metrics(
            logits=out["logits_vp"],
            vad=batch["vad"],
            vad_label=batch["vad_label"],
            min_context_frames=50,
            k=5,
        )
        m["loss"] = loss
        return m

    def test_step(self, batch, batch_idx, **kwargs):
        batch_size = batch["waveform"].shape[0]
        loss, out, _, batch_size = self.shared_step(batch)
        self.log("test_loss", loss["total"], batch_size=batch_size)
        self.log("test_loss_vp", loss["vp"], batch_size=batch_size)
        if self.conf["tier1"]["num_layers"] > 0:
            self.log("loss_ar", loss["ar"], batch_size=batch_size)

        m = self.projection_head.prepare_metrics(
            logits=out["logits_vp"],
            vad=batch["vad"],
            vad_label=batch["vad_label"],
            min_context_frames=50,
            k=5,
        )
        m["loss"] = loss
        return m

    @staticmethod
    def add_model_specific_args(parent_parser):
        """argparse arguments for SoSIModel (based on yaml-config)"""
        parser = parent_parser.add_argument_group("ULMProjection")
        parser.add_argument("--conf", default=None, type=str)
        parser.add_argument("--dont_log_model", action="store_true")

        # A workaround for OmegaConf + WandB-Sweeps
        default_conf = ULMProjection.load_config(format=None)
        parser = OmegaConfArgs.add_argparse_args(parser, default_conf)
        return parent_parser


def ani_debug():

    from argparse import ArgumentParser
    from datasets_turntaking.dm_dialog_audio import (
        DialogAudioDM,
        DialogIPU,
        get_dialog_audio_datasets,
        print_dm,
    )

    parser = ArgumentParser()
    parser = DialogAudioDM.add_data_specific_args(parser)
    parser = ULMProjection.add_model_specific_args(parser)
    args = parser.parse_args()
    data_conf = DialogAudioDM.load_config(path=args.data_conf, args=args)
    # data_conf["dataset"]["type"] = "sliding"
    print_dm(data_conf, args)

    data_conf = DialogAudioDM.load_config(path=args.data_conf, args=args)
    val_hf_dataset = get_dialog_audio_datasets(
        datasets=data_conf["dataset"]["datasets"], split="val"
    )
    sample_dset = DialogIPU(
        dataset=val_hf_dataset,
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hop_time=data_conf["dataset"]["vad_hop_time"],
        vad_bin_sizes=data_conf["dataset"]["vad_bin_sizes"],
    )

    diter = iter(sample_dset)

    conf = ULMProjection.load_config(path=args.conf, args=args)
    model = ULMProjection(conf)

    batch = next(diter)

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    from argparse import ArgumentParser
    from datasets_turntaking.dm_dialog_audio import DialogAudioDM, print_dm
    from conv_ssl.utils import count_parameters

    parser = ArgumentParser()
    parser = DialogAudioDM.add_data_specific_args(parser)
    parser = ULMProjection.add_model_specific_args(parser)
    args = parser.parse_args()
    data_conf = DialogAudioDM.load_config(path=args.data_conf, args=args)
    # data_conf["dataset"]["type"] = "sliding"
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
        batch_size=4,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup()
    diter = iter(dm.val_dataloader())

    conf = ULMProjection.load_config(path=args.conf, args=args)
    conf["tier1"]["dim"] = 64
    conf["tier2"]["dim"] = conf["tier1"]["dim"]
    conf["tier1"]["num_layers"] = 1
    conf["tier1"]["num_heads"] = 1
    conf["tier2"]["num_layers"] = 1
    conf["tier2"]["num_heads"] = 1
    conf["vad_class_prediction"]["regression"] = True
    model = ULMProjection(conf)
    name = model.run_name  # Name the run e.g. hubert_44_41
    print("-" * 60)
    print(f"Model Name: {name}")
    print("Base: ", args.conf)
    print("PARAMETERS: ", count_parameters(model))
    print()
    batch = next(diter)
    print("waveform: ", tuple(batch["waveform"].shape))
    print("vad: ", tuple(batch["vad"].shape))
    print("vad_history: ", tuple(batch["vad_history"].shape))
    print("vad_label: ", tuple(batch["vad_label"].shape))
    print("-" * 60)
    loss, out, batch, batch_size = model.shared_step(batch)
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    # # regression
    # model = ULMProjection.load_from_checkpoint(
    #     "runs/conv_ssl/ULMProjection/ULMProjection_3ptfboyq/epoch=10-val_loss=0.82926.ckpt"
    # )
    # loss, out, batch, batch_size = model.shared_step(batch)
    # # logits -> sigmoid -> probs -> round -> binary
    # # vp = out["logits_vp"].sigmoid().round()
    # # vp = (
    # #     model.ulm_projection.vad_projection_codebook.onehot_to_idx(vp)
    # #     .float()
    # #     .unsqueeze(-1)
    # # )
    # m = model.vad_projection_codebook.prepare_metrics(
    #     out, batch, min_context_frames=min_context_frames, k=1, cpu=True
    # )
    #
    # for k, v in m.items():
    #     print(f"{k}: {v}")
