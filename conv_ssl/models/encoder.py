from os.path import join

import torch
import torch.nn as nn
import einops

from conv_ssl.models.cpc_base_model import load_CPC
from conv_ssl.utils import load_config, repo_root

DEFAULT_CONFIG = join(repo_root(), "conv_ssl/config/model.yaml")
MODEL_HZ = {"cpc": 100}
MODEL_DIM = {"cpc": 256}


class Encoder(nn.Module):
    """
    Encoder: waveform -> h
    pretrained: default='cpc'

    A simpler version of the Encoder
    check paper (branch) version to see other encoders...
    """

    def __init__(self, conf, freeze=True):
        super().__init__()
        self.conf = conf
        self.name = conf["name"]
        self.frame_hz = conf["frame_hz"]
        self.encoder_layer = conf["output_layer"]
        self.encoder = load_CPC()
        self.output_dim = 256
        if freeze:
            self.freeze()

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        print(f"Froze {self.__class__.__name__}!")

    def encode(self, waveform):
        if waveform.ndim < 3:
            waveform = waveform.unsqueeze(1)  # channel dim

        # Backwards using only the encoder encounters:
        # ---------------------------------------------------
        # RuntimeError: one of the variables needed for gradient computation
        # has been modified by an inplace operation:
        # [torch.FloatTensor [4, 256, 1000]], which is output 0 of ReluBackward0, is at version 1;
        # expected version 0 instead. Hint: enable anomaly detection to find
        # the operation that failed to compute its gradient, with
        # torch.autograd.set_detect_anomaly(True).
        z = self.encoder.gEncoder(waveform)  # .permute(0, 2, 1)
        z = einops.rearrange(z, "b c n -> b n c")

        # However, if we feed through gAR we do not encounter that problem...
        if self.encoder_layer > 0:
            z = self.encoder.gAR(z)
        return z

    def forward(self, waveform):
        z = self.encode(waveform)
        return {"z": z}
