from os.path import join
from typing import Any, Dict
import matplotlib as mpl

import torch
import pytorch_lightning as pl

mpl.use("Agg")

from conv_ssl.models import ProjectionModel, EncoderPretrained, CHECKPOINTS
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
        self.ulm_projection = ProjectionModel(self.conf)

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
            encoder = EncoderPretrained(conf, load=load)
            # update dimension of encoder
            self.conf["encoder"]["dim"] = encoder.conf["encoder"]["dim"]
            return encoder
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

    def summary(self):
        s = "ULMProjection\n"
        s += "Encoder\n"
        s += f"\ttype: {self.conf['encoder']['type']}\n"
        s += f"\toutput_layer: {self.conf['encoder']['output_layer']}\n"
        s += f"\tHz: {self.conf['encoder']['frame_hz']}\n"
        s += f"\tquantizer n_codes: {self.conf['quantizer']['n_codes']}\n"
        if self.ulm_projection.tier1 is not None:
            s += "Tier1\n"
            s += f"\tnum_layers: {self.conf['tier1']['num_layers']}\n"
            s += f"\tnum_heads: {self.conf['tier1']['num_heads']}\n"
            s += f"\tdim: {self.conf['tier1']['dim']}\n"
        if self.ulm_projection.tier2 is not None:
            s += "Tier2\n"
            s += f"\tnum_layers: {self.conf['tier2']['num_layers']}\n"
            s += f"\tnum_heads: {self.conf['tier2']['num_heads']}\n"
            s += f"\tdim: {self.conf['tier2']['dim']}\n"
        s += "Head\n"
        s += f"\tregression: {self.conf['vad_class_prediction']['regression']}\n"
        return s

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
        if input_ids is not None:
            out["enc_out"] = input_ids
        else:
            enc_out = self.encoder(waveform)
            if "q_idx" in enc_out:
                out["enc_out"] = enc_out["q_idx"]
            else:
                out["enc_out"] = enc_out["z"]

        # logits_ar & logits_vp
        o = self.ulm_projection(out["enc_out"], vad=vad, vad_history=vad_history)
        out.update(o)
        return out

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

        vad_history = batch.get("vad_history", None)
        input_ids = batch.get("q", None)
        waveform = batch.get("waveform", None)
        out = self(
            waveform=waveform,
            input_ids=input_ids,
            vad=batch["vad"],
            vad_history=vad_history,
        )
        # update batch to match size
        # some encoders (wav2vec, vq_wav2vec) drops 2 frames on 10sec audio
        # some encoders (wavlm_base, hubert) drops 1 frames (after downsampling) on 10sec audio
        n_frames = out["logits_vp"].shape[1]
        batch["vad"] = batch["vad"][:, :n_frames]
        batch["vad_label"] = batch["vad_label"][:, :n_frames]

        batch_size = out["enc_out"].shape[0]
        loss = self.ulm_projection.calc_losses(
            out,
            vad_labels=batch["vad_label"],
            input_ids=out["enc_out"],
            reduction=reduction,
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
        if self.ulm_projection.tier1 is not None:
            self.log("loss_ar", loss["ar"], batch_size=batch_size)
        return {"loss": loss["total"]}

    def validation_step(self, batch, batch_idx, **kwargs):
        loss, out, batch, batch_size = self.shared_step(batch)
        self.log("val_loss", loss["total"], batch_size=batch_size)
        self.log("val_loss_vp", loss["vp"], batch_size=batch_size)
        if self.ulm_projection.tier1 is not None:
            self.log("loss_ar", loss["ar"], batch_size=batch_size)

        out["batch"] = batch
        return {"loss": loss, "outputs": out}

    def test_step(self, batch, batch_idx, **kwargs):
        batch_size = batch["waveform"].shape[0]
        loss, out, _, batch_size = self.shared_step(batch)
        self.log("test_loss", loss["total"], batch_size=batch_size)
        self.log("test_loss_vp", loss["vp"], batch_size=batch_size)
        if self.ulm_projection.tier1 is not None:
            self.log("loss_ar", loss["ar"], batch_size=batch_size)
        return {"loss": loss, "outputs": out}

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


if __name__ == "__main__":
    from argparse import ArgumentParser
    from datasets_turntaking import DialogAudioDM
    from conv_ssl.utils import count_parameters

    parser = ArgumentParser()
    parser = DialogAudioDM.add_data_specific_args(parser)
    parser = ULMProjection.add_model_specific_args(parser)
    args = parser.parse_args()
    data_conf = DialogAudioDM.load_config(path=args.data_conf, args=args)
    # data_conf["dataset"]["type"] = "sliding"
    DialogAudioDM.print_dm(data_conf, args)

    conf = ULMProjection.load_config(path=args.conf, args=args)
    conf["encoder"]["type"] = "vq_wav2vec"
    conf["encoder"]["output_layer"] = 0
    conf["quantizer"]["n_codes"] = 0
    conf["tier1"]["num_layers"] = 0
    conf["tier2"]["num_layers"] = 1
    conf["tier2"]["num_heads"] = 4
    conf["tier2"]["dim"] = 256
    model = ULMProjection(conf)
    name = model.run_name  # Name the run e.g. hubert_44_41
    print("-" * 60)
    print(model.summary())
    print("Base: ", args.conf)
    print("PARAMETERS: ", count_parameters(model))
    print()

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
        vad_hz=model.encoder.frame_hz,
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
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")
    print("-" * 50)
    loss, out, batch, batch_size = model.shared_step(batch)
