import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from conv_ssl.models.cpc_base_model import load_CPC
from conv_ssl.models.cnn import CConv1d, LayerNorm


def get_cnn_layer(dim, kernel, stride, dilation, activation):
    layers = [Rearrange("b t d -> b d t")]
    for k, s, d in zip(kernel, stride, dilation):
        layers.append(CConv1d(dim, dim, kernel_size=k, stride=s, dilation=d))
        layers.append(LayerNorm(dim))
        layers.append(getattr(torch.nn, activation)())
    layers.append(Rearrange("b d t -> b t d"))
    return nn.Sequential(*layers)


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
        self.output_dim = self.encoder.gEncoder.conv4.out_channels

        if conf.get("downsample", False):
            down = conf["downsample"]
            self.downsample = get_cnn_layer(
                dim=self.output_dim,
                kernel=down["kernel"],
                stride=down["stride"],
                dilation=down["dilation"],
                activation=down["activation"],
            )
        else:
            self.downsample = nn.Identity()

        if freeze:
            self.freeze()

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        print(f"Froze {self.__class__.__name__}!")

    def unfreeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        print(f"Trainable {self.__class__.__name__}!")

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
        z = self.downsample(z)
        return {"z": z}


def _test_encoder(config_name):
    from conv_ssl.utils import load_hydra_conf

    conf = load_hydra_conf(config_name=config_name)
    econf = conf["model"]["encoder"]
    enc = Encoder(econf, freeze=econf["freeze"])
    x = torch.rand((4, econf["sample_rate"]))
    out = enc(x)
    z = out["z"]
    print("Config: ", config_name)
    print("x: ", tuple(x.shape))
    print("z: ", tuple(z.shape))


if __name__ == "__main__":
    _test_encoder("model/discrete")
    _test_encoder("model/discrete_20hz")
    _test_encoder("model/discrete_50hz")
