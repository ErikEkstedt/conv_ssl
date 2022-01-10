import pytest
import torch
from os.path import join
from conv_ssl.models import EncoderPretrained
from conv_ssl.utils import repo_root


@pytest.mark.models
@pytest.mark.encoder
@pytest.mark.parametrize("name", ["hubert_base", "wav2vec2_base", "wavlm_base+", "cpc"])
def test_encoder(name):
    conf_path = join(repo_root(), "conv_ssl/config")
    if name == "hubert_base":
        conf_path = join(conf_path, "encoder_hubert.yaml")
    elif name == "wav2vec2_base":
        conf_path = join(conf_path, "encoder_wav2vec2.yaml")
    elif name == "wavlm_base+":
        conf_path = join(conf_path, "encoder_wavlm.yaml")
    else:
        conf_path = join(conf_path, "encoder_cpc.yaml")

    conf = EncoderPretrained.load_config(conf_path)
    conf["quantizer"]["vector_path"] = None
    model = EncoderPretrained(conf)

    # extract the representation of last layer
    wav_input_16khz = torch.randn(1, 16000)
    z = model.encode(wav_input_16khz)

    output_shape = (1, 49, 768)
    if name == "cpc":
        output_shape = (1, 100, 256)

    assert tuple(z.shape) == output_shape, f"{name} output_shape != {output_shape}"
