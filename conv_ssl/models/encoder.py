from os.path import join

import torch
import torch.nn as nn
import einops

from conv_ssl.models.pretrained_encoders import load_pretrained_encoder
from conv_ssl.models.kmean import KMeanEmbedding
from conv_ssl.utils import load_config, repo_root

DEFAULT_CONFIG = join(repo_root(), "conv_ssl/config/ulm.yaml")
MODEL_HZ = {
    "hubert_base": 50,
    "wav2vec2": 50,
    "wavlm_base": 50,
    "wavlm_base+": 50,
    "wav2vec": 100,
    "vq_wav2vec": 100,
    "cpc": 100,
}


class EncoderPretrained(nn.Module):
    """
    Encoder (pretrained, default='hubert_base')
    includes a pretrained encoder and pretrained quantized feature embeddings (kmeans)
    """

    def __init__(self, conf, load=False, freeze=True):
        super().__init__()
        self.conf = {"encoder": conf["encoder"]}
        self.conf["quantizer"] = conf["quantizer"]
        self.frame_hz = conf["encoder"]["frame_hz"]

        # Encoder: Hubert (torchaudio) `output_layer` defines which representations to use
        self.encoder = load_pretrained_encoder(conf["encoder"]["type"])
        self.encoder_layer = conf["encoder"]["output_layer"]

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

    def fix_batch_hz(self, batch):
        """
        handle different Hz. VadPred is 100hz (10ms) and hubert/wav2vec2 is 50hz (20ms)
        hubert_base is twice as slow as vadpred-features
        so we use the last frame from vadpred for each step of hubert
        e.g. x is the chosen label frame
        -------------------------
        vadpred :   | |x| |x| |x|
        frame   :   |0|1|2|3|4|5|
        -------------------------
        cpc     :   |0|1|2|3|4|5|
        hubert  :   | 0 | 1 | 2 |
        wavlm   :   | 0 | 1 | 2 |
        wav2vec2:   | 0 | 1 | 2 |
        """
        if MODEL_HZ[self.name] == 50:
            batch["vad"] = self.hz_100_to_50(batch["vad"])[:, :-1]
            batch["vad_label"] = self.hz_100_to_50(batch["vad_label"])[:, :-1]
            if "vad_history" in batch:
                batch["vad_history"] = self.hz_100_to_50(batch["vad_history"])[:, :-1]
        return batch

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        if self.quantizer is not None:
            for p in self.quantizer.parameters():
                p.requires_grad_(False)
        print(f"Froze {self.__class__.__name__}!")

    def hz_100_to_50(self, x):
        """Return every other frame to go from 100Hz -> 50Hz"""
        return x[:, 1::2]

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
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        z_c, z_enc, _ = self.encoder(waveform, label=None)  # c, z, label
        if self.encoder_layer == 0:
            z = z_enc
        else:
            z = z_c
        return z

    def encode(self, waveform):
        if self.conf["encoder"]["type"] in ["hubert_base", "wav2vec2_base"]:
            z = self._hubert_wav2vec2(waveform)
        elif self.conf["encoder"]["type"] in ["wavlm_base", "wavlm_base+"]:
            z = self._wavlm(waveform)
        elif self.conf["encoder"]["type"] in ["vq_wav2vec", "wav2vec"]:
            z = self._wav2vec(waveform)
        elif self.conf["encoder"]["type"] == "cpc":
            z = self._cpc(waveform)
        return z

    def forward(self, waveform):
        z = self.encode(waveform)
        ret = {"z": z}
        if self.quantizer is not None:
            ret["q"], ret["q_idx"] = self.quantizer(z)
        return ret


if __name__ == "__main__":

    name = "wavlm_base+"
    conf_path = join(repo_root(), "conv_ssl/config")
    if name == "hubert_base":
        conf_path = join(conf_path, "ulm_hubert.yaml")
    elif name == "wav2vec":
        conf_path = join(conf_path, "ulm_wav2vec.yaml")
    elif name == "vq_wav2vec":
        conf_path = join(conf_path, "ulm_vq_wav2vec.yaml")
    elif name == "wav2vec2_base":
        conf_path = join(conf_path, "ulm_wav2vec2.yaml")
    elif name == "wavlm_base+":
        conf_path = join(conf_path, "ulm_wavlm.yaml")
    elif name == "cpc+":
        conf_path = join(conf_path, "ulm_wavlm.yaml")
    else:
        assert False, f"{name} is not found"

    conf = EncoderPretrained.load_config(conf_path)
    conf["quantizer"]["vector_path"] = None
    model = EncoderPretrained(conf)

    wav_input_16khz = torch.randn(1, 16000)
