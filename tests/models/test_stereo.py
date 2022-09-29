import pytest

import torch
from conv_ssl.model import ProjectionModelStereo
from conv_ssl.utils import load_hydra_conf


@pytest.mark.stereo
@pytest.mark.parametrize("config_name", ["model/vap_stereo"])
def test_cpc_encoder(config_name):
    conf = load_hydra_conf(config_name=config_name)
    model = ProjectionModelStereo(conf["model"])

    # extract the representation of last layer
    stereo_waveform = torch.randn(1, 2, conf["model"]["encoder"]["sample_rate"])

    if torch.cuda.is_available():
        stereo_waveform = stereo_waveform.to("cuda")
        model.to("cuda")

    z = model(stereo_waveform)
