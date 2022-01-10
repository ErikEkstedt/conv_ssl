import pytest
import torch

from conv_ssl.models.pretrained_encoders import load_wavlm


@pytest.mark.models
def test_wavlm_base():
    model = load_wavlm("wavlm_base")

    # extract the representation of last layer
    wav_input_16khz = torch.randn(1, 16000)
    rep = model.extract_features(wav_input_16khz)[0]
    assert tuple(rep.shape) == (
        1,
        49,
        768,
    ), "Wrong output shape: output_shape != (1, 49, 768)"


@pytest.mark.models
def test_wavlm_base_plus():
    model = load_wavlm("wavlm_base+")

    # extract the representation of last layer
    wav_input_16khz = torch.randn(1, 16000)
    rep = model.extract_features(wav_input_16khz)[0]
    assert tuple(rep.shape) == (
        1,
        49,
        768,
    ), "Wrong output shape: output_shape != (1, 49, 768)"
