from os.path import join

import torch
import torch.nn as nn
import einops

from conv_ssl.models.pretrained_encoders import load_pretrained_encoder
from conv_ssl.models.kmean import KMeanEmbedding
from conv_ssl.utils import load_config, repo_root

DEFAULT_CONFIG = join(repo_root(), "conv_ssl/config/model.yaml")
MODEL_HZ = {
    "hubert_base": 50,
    "wav2vec2_base": 50,
    "wavlm_base": 50,
    "wavlm_base+": 50,
    "wav2vec": 100,
    "vq_wav2vec": 100,
    "cpc": 100,
}

MODEL_DIM = {
    "hubert_base": 768,
    "wav2vec2_base": 768,
    "wavlm_base": 768,
    "wavlm_base+": 768,
    "wav2vec": 512,
    "vq_wav2vec": 512,
    "cpc": 256,
}


class Downsample(nn.Module):
    def __init__(self, dim=512, n_frames=5, type="mean", dimension=-1) -> None:
        super().__init__()
        self.dim = dim
        self.dimension = dimension
        self.n_frames = n_frames
        self.type = type
        self.net = self._build()

    def __repr__(self):
        s = "Downsampler(\n"
        s += f"\ttype: {self.type}\n"
        s += f"\tn_frames: {self.n_frames}\n"
        s += ")"
        return s

    def _build(self):
        if self.type == "cnn":
            net = nn.Conv1d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=self.n_frames,
                stride=self.n_frames,
            )
        elif self.type == "maxpool":
            net = nn.MaxPool1d(kernel_size=self.n_frames, stride=self.n_frames)
        else:
            net = None
        return net

    def forward(self, x):
        if self.type == "mean":
            x = x.unfold(
                dimension=self.dimension, step=self.n_frames, size=self.n_frames
            ).mean(dim=-1)
        elif self.type == "last":
            x = x.unfold(
                dimension=self.dimension, step=self.n_frames, size=self.n_frames
            )[..., -1]
        else:
            x = einops.rearrange(x, "b n c -> b c n")
            x = self.net(x)
            x = einops.rearrange(x, "b c n -> b n c")
        return x


class EncoderPretrained(nn.Module):
    """
    Encoder (pretrained, default='hubert_base')
    includes a pretrained encoder and pretrained quantized feature embeddings (kmeans)
    """

    def __init__(self, conf, load=False, freeze=True):
        super().__init__()
        self.conf = self.encoder_conf(conf)
        self.frame_hz = conf["encoder"]["frame_hz"]

        # Encoder: Hubert (torchaudio) `output_layer` defines which representations to use
        self.encoder = load_pretrained_encoder(conf["encoder"]["type"])
        self.encoder_layer = conf["encoder"]["output_layer"]

        # Downsampling
        self.downsampler = None
        if (
            conf["encoder"]["downsample"]
            and conf["encoder"]["target_frame_hz"] != self.frame_hz
            and conf["encoder"]["target_frame_hz"] != 0
        ):
            dim = 728
            if conf["encoder"]["type"] in ["wav2vec", "vq_wav2vec"]:
                dim = 512
            n_frames = self.frame_hz // conf["encoder"]["target_frame_hz"]
            self.frame_hz = conf["encoder"]["target_frame_hz"]
            self.downsampler = Downsample(
                dim=dim,
                n_frames=n_frames,
                type=conf["encoder"]["downsample"],
                dimension=1,
            )

        # Vector quantization
        self.quantizer = None
        if conf["quantizer"]["n_codes"] > 0:
            self.quantizer = KMeanEmbedding(
                k=conf["quantizer"]["n_codes"],
                dim=conf["encoder"]["dim"],
                vectors=None if load else conf["quantizer"]["vector_path"],
            )

        if freeze:
            self.freeze()

    def encoder_conf(self, conf):
        enc_conf = {"encoder": conf["encoder"]}
        enc_conf["quantizer"] = conf["quantizer"]
        enc_conf["encoder"]["dim"] = MODEL_DIM[conf["encoder"]["type"]]
        enc_conf["encoder"]["frame_hz"] = MODEL_HZ[conf["encoder"]["type"]]
        return enc_conf

    @property
    def name(self):
        return self.conf["encoder"]["type"]

    @staticmethod
    def default_config_path():
        return DEFAULT_CONFIG

    @staticmethod
    def load_config(path=None, args=None, format="dict"):
        if path is None:
            path = EncoderPretrained.default_config_path()
        return load_config(path, args=args, format=format)

    @staticmethod
    def load_state_dict(path):
        return torch.load(path)

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        if self.quantizer is not None:
            for p in self.quantizer.parameters():
                p.requires_grad_(False)
        print(f"Froze {self.__class__.__name__}!")

    def get_embeddings(self, waveform):
        return self(waveform)["q_idx"]

    def _hubert_wav2vec2(self, waveform):
        return self.encoder.extract_features(waveform)[0][self.encoder_layer]

    def _wavlm(self, waveform):
        return self.encoder.extract_features(waveform, output_layer=self.encoder_layer)[
            0
        ]

    def _wav2vec(self, waveform):
        z = self.encoder.feature_extractor(waveform)
        if self.conf["encoder"]["type"] == "vq-wav2vec":
            if self.encoder_layer > 0:
                z = self.encoder.feature_aggregator(z)
                z = einops.rearrange(z, "... c t -> ... t c")
            else:
                _, z = self.encoder.vector_quantizer.forward_idx(z)
        else:
            if self.encoder_layer > 0:
                z = self.encoder.feature_aggregator(z)
            z = einops.rearrange(z, "... c t -> ... t c")
        return z

    def _cpc(self, waveform):
        if waveform.ndim < 3:
            waveform = waveform.unsqueeze(1)  # channel dim
        c, z, _ = self.encoder(waveform, None)

        if self.encoder_layer > 0:
            return c
        else:
            return z

    def encode(self, waveform):
        z = None
        if self.conf["encoder"]["type"] in ["hubert_base", "wav2vec2_base"]:
            z = self._hubert_wav2vec2(waveform)
        elif self.conf["encoder"]["type"] in ["wavlm_base", "wavlm_base+"]:
            z = self._wavlm(waveform)
        elif self.conf["encoder"]["type"] in ["vq_wav2vec", "wav2vec"]:
            z = self._wav2vec(waveform)
        elif self.conf["encoder"]["type"] == "cpc":
            z = self._cpc(waveform)

        if self.downsampler is not None:
            z = self.downsampler(z)
        return z

    def forward(self, waveform):
        z = self.encode(waveform)
        ret = {"z": z}
        if self.quantizer is not None:
            ret["q"], ret["q_idx"] = self.quantizer(z)
        return ret


if __name__ == "__main__":

    # name = "model.yaml"
    # conf_path = join(repo_root(), "conv_ssl/config")
    # if name == "hubert_base":
    #     conf_path = join(conf_path, "ulm_hubert.yaml")
    # elif name == "wav2vec":
    #     conf_path = join(conf_path, "ulm_wav2vec.yaml")
    # elif name == "vq_wav2vec":
    #     conf_path = join(conf_path, "ulm_vq_wav2vec.yaml")
    # elif name == "wav2vec2_base":
    #     conf_path = join(conf_path, "ulm_wav2vec2.yaml")
    # elif name == "wavlm_base+":
    #     conf_path = join(conf_path, "ulm_wavlm.yaml")
    # elif name == "cpc+":
    #     conf_path = join(conf_path, "ulm_wavlm.yaml")
    # else:
    #     assert False, f"{name} is not found"

    conf = EncoderPretrained.load_config()
    conf["encoder"]["type"] = "wav2vec"
    conf["encoder"]["target_frame_hz"] = 20
    conf["encoder"]["downsample"] = "mean"
    conf["quantizer"]["vector_path"] = None
    model = EncoderPretrained(conf)
    n = count_parameters(model, learnable=True)
    print("parameters: ", n)

    sr = 16000
    wav_input_16khz = torch.randn(1, sr * 10)

    o = model(wav_input_16khz)
    print("o['z']: ", tuple(o["z"].shape))

    from conv_ssl.utils import count_parameters
