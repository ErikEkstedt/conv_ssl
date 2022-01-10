import pytest
import torch

from conv_ssl.models.pretrained_encoders import load_CPC


@pytest.mark.models
def test_cpc():
    model = load_CPC()

    # extract the representation of last layer
    wav_input_16khz = torch.randn(1, 1, 16000)
    z_c, z_enc, _ = model(wav_input_16khz, label=None)  # c, z, label

    assert tuple(z_c.shape) == (
        1,
        100,
        256,
    ), "Wrong z_c (c) shape: output_shape != (1, 49, 256)"
    assert tuple(z_enc.shape) == (
        1,
        100,
        256,
    ), "Wrong z_enc (z) shape: output_shape != (1, 100, 256)"
