import pytest

import torch
from conv_ssl.models.transformer import TransformerStereoLayer, GPTStereo


@pytest.fixture(scope="module")
def layer():
    layer = TransformerStereoLayer(
        dim=256, ffn_dim=512, num_heads=8, cross_attention=True
    )
    return layer


@pytest.mark.transformer
@pytest.mark.stereo
def test_stereo_layer_dim(layer):
    x1 = torch.rand((4, 20, layer.dim))
    x2 = torch.rand((4, 20, layer.dim))
    z1, z2 = layer(x1, x2)
    assert z1.shape == x1.shape, f"z1 {tuple(z1.shape)} != x1 {tuple(x1.shape)}"
    assert z2.shape == x2.shape, f"z2 {tuple(z2.shape)} != x2 {tuple(x2.shape)}"


@pytest.mark.transformer
@pytest.mark.stereo
def test_stereo_grads(layer):
    # Z2 -> X1 grads
    x1 = torch.rand((4, 20, layer.dim)).requires_grad_(True)
    x2 = torch.rand((4, 20, layer.dim))
    z1, z2 = layer(x1, x2)
    z2.sum().backward()
    n = x1.grad.abs().sum()
    assert n > 0, "(single) No gradient from z2 -> x1"

    # Z1 -> X2 grads
    x1 = torch.rand((4, 20, layer.dim))
    x2 = torch.rand((4, 20, layer.dim)).requires_grad_(True)
    z1, z2 = layer(x1, x2)
    z1.sum().backward()
    n = x2.grad.abs().sum()
    assert n > 0, "(single) No gradient from z1 -> x2"

    # Z1 -> X2 grads
    x1 = torch.rand((4, 20, layer.dim)).requires_grad_(True)
    x2 = torch.rand((4, 20, layer.dim)).requires_grad_(True)
    z1, z2 = layer(x1, x2)
    z1.sum().backward()
    n1 = x1.grad.abs().sum()
    n2 = x2.grad.abs().sum()
    assert n1 > 0, "(both) No gradient from z1 -> x1"
    assert n2 > 0, "(both) No gradient from z1 -> x2"


@pytest.mark.transformer
@pytest.mark.stereo
def test_gpt_stereo_dim():

    model = GPTStereo(dim=256, dff_k=3, num_layers=4, num_heads=8)

    # Z2 -> X1 grads
    x1 = torch.rand((4, 20, model.dim)).requires_grad_(True)
    x2 = torch.rand((4, 20, model.dim)).requires_grad_(True)
    z = model(x1, x2)
    z.sum().backward()
    n1 = x1.grad.abs().sum()
    n2 = x1.grad.abs().sum()
    assert z.shape == x1.shape == x2.shape, "Different shapes"
    assert n1 > 0, "No gradient from z -> x1"
    assert n2 > 0, "No gradient from z -> x2"
