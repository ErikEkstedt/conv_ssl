import pytest
import torch
from os.path import join
from conv_ssl.models import Encoder
from conv_ssl.utils import repo_root


@pytest.mark.models
@pytest.mark.encoder
@pytest.mark.cpc
def test_cpc_encoder():
    name = "cpc"
    conf = Encoder.load_config()
    conf["encoder"]["type"] = name
    conf["encoder"]["output_layer"] = 6
    conf["quantizer"]["vector_path"] = None
    model = Encoder(conf)
    # extract the representation of last layer
    wav_input_16khz = torch.randn(1, 16000)
    if torch.cuda.is_available():
        wav_input_16khz = wav_input_16khz.to("cuda")
        model.to("cuda")
    z = model.encode(wav_input_16khz)


@pytest.mark.models
@pytest.mark.encoder
@pytest.mark.wav2vec
@pytest.mark.parametrize("name", ["wav2vec", "vq_wav2vec"])
def test_wav2vec_encoder(name):
    conf = Encoder.load_config()
    conf["encoder"]["type"] = name
    conf["encoder"]["output_layer"] = 6
    conf["quantizer"]["vector_path"] = None
    model = Encoder(conf)
    # extract the representation of last layer
    wav_input_16khz = torch.randn(1, 16000)
    if torch.cuda.is_available():
        wav_input_16khz = wav_input_16khz.to("cuda")
        model.to("cuda")
    z = model.encode(wav_input_16khz)


@pytest.mark.models
@pytest.mark.encoder
@pytest.mark.non_causal
@pytest.mark.parametrize(
    "name",
    ["hubert_base", "wav2vec2_base", "wavlm_base+"],
)
def test_non_causal_encoder(name):
    conf = Encoder.load_config()
    conf["encoder"]["type"] = name
    conf["encoder"]["output_layer"] = 6
    conf["quantizer"]["vector_path"] = None
    model = Encoder(conf)

    # extract the representation of last layer
    wav_input_16khz = torch.randn(1, 16000)

    if torch.cuda.is_available():
        wav_input_16khz = wav_input_16khz.to("cuda")
        model.to("cuda")
    z = model.encode(wav_input_16khz)
