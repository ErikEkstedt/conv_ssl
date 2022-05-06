import pytest

import torch
from conv_ssl.models import Encoder
from conv_ssl.utils import load_hydra_conf


@pytest.mark.encoder
@pytest.mark.parametrize("config_name", ["model/discrete", "model/discrete_20hz"])
def test_cpc_encoder(config_name):
    conf = load_hydra_conf(config_name=config_name)
    enc_conf = conf["model"]["encoder"]
    model = Encoder(enc_conf)

    # extract the representation of last layer
    wav_input_16khz = torch.randn(1, enc_conf["sample_rate"])

    if torch.cuda.is_available():
        wav_input_16khz = wav_input_16khz.to("cuda")
        model.to("cuda")

    z = model.encode(wav_input_16khz)
    assert tuple(z.shape) == (1, 100, 256), "shape mismatch"
