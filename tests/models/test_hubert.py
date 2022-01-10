import pytest
import torch

from conv_ssl.models.pretrained_encoders import load_hubert_base_custom


@pytest.mark.models
def test_hubert_base():
    model = load_hubert_base_custom()

    # extract the representation of last layer
    wav_input_16khz = torch.randn(1, 16000)
    encoder_layer = 6
    rep = model.extract_features(wav_input_16khz)[0][encoder_layer]
    assert tuple(rep.shape) == (
        1,
        49,
        768,
    ), "Wrong output shape: output_shape != (1, 49, 768)"
