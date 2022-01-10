from os.path import join

import torch
import torch.nn as nn

from conv_ssl.models.pretrained_encoders import load_pretrained_encoder
from conv_ssl.models.kmean import KMeanEmbedding
from conv_ssl.utils import load_config, repo_root

DEFAULT_CONFIG = join(repo_root(), "conv_ssl/config/encoder.yaml")


class EncoderPretrained(nn.Module):
    """
    Encoder (pretrained, default='hubert_base')
    includes a pretrained encoder and pretrained quantized feature embeddings (kmeans)
    """

    def __init__(self, conf, load=False, freeze=True):
        super().__init__()
        self.conf = conf
        self.frame_hz = conf["encoder"]["frame_hz"]

        # Encoder: Hubert (torchaudio) `output_layer` defines which representations to use
        self.encoder = load_pretrained_encoder(conf["encoder"]["type"])
        self.encoder_layer = conf["encoder"]["output_layer"]

        # Vector quantization
        self.quantizer = KMeanEmbedding(
            k=conf["quantizer"]["n_codes"],
            dim=conf["quantizer"]["dim"],
            vectors=None if load else conf["quantizer"]["vector_path"],
        )

        if freeze:
            self.freeze()

    @property
    def name(self):
        return self.conf["encoder"]["type"]

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        for p in self.quantizer.parameters():
            p.requires_grad_(False)
        print(f"Froze {self.__class__.__name__}!")

    def hz_100_to_50(self, x):
        """Return every other frame to go from 100Hz -> 50Hz"""
        return x[:, 1::2]

    def get_embeddings(self, waveform):
        return self(waveform)["q_idx"]

    def encode(self, waveform):
        if self.conf["encoder"]["type"] in ["hubert_base", "wav2vec2_base"]:
            z = self.encoder.extract_features(waveform)[0][self.encoder_layer]
        else:
            if waveform.ndim == 2:
                waveform = waveform.unsqueeze(1)
            z_c, z_enc, _ = self.encoder(waveform, label=None)  # c, z, label
            if self.encoder_layer == 0:
                z = z_enc
            else:
                z = z_c
        return z

    def forward(self, waveform):
        z = self.encode(waveform)
        q, q_idx = self.quantizer(z)
        return {"q": q, "q_idx": q_idx, "z": z}

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


def test(name):
    from datasets_turntaking.dm_dialog_audio import quick_load_dm
    from conv_ssl.utils import count_parameters

    dm = quick_load_dm(batch_size=4, num_workers=0)
    dm.setup()

    # Try out a dataloder with the awesome iterable dataset
    dloader = dm.val_dataloader()
    diter = iter(dloader)

    conf_path = join(repo_root(), "conv_ssl/config")
    if name == "hubert_base":
        conf_path = join(conf_path, "encoder.yaml")
    elif name == "cpc":
        conf_path = join(conf_path, "encoder_cpc.yaml")
    elif name == "wav2vec2_base":
        conf_path = join(conf_path, "encoder_wav2vec2.yaml")
    else:
        raise NotImplementedError(f"Name: {name}")

    conf = EncoderPretrained.load_config(conf_path)
    conf["quantizer"]["vector_path"] = None
    model = EncoderPretrained(conf)
    n = count_parameters(model, as_string=True, learnable=False)
    print(f"{model.name.upper()}: ", n)

    batch = next(diter)
    print("batch['waveform']: ", tuple(batch["waveform"].shape))

    z = model.encode(batch["waveform"])
    print("z: ", tuple(z.shape))


if __name__ == "__main__":
    import sys

    name = "hubert_base"
    if len(sys.argv) > 1:
        name = sys.argv[1]
        del sys.argv[1]
    test(name)
