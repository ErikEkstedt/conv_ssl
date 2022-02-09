import pytest

import torch
from conv_ssl.model import VPModel
from conv_ssl.utils import everything_deterministic
from conv_ssl.evaluation import load_dm, to_device

everything_deterministic()

DURATION = 10
SAMPLE_RATE = 16000
FRAME_HZ = 100
BATCH_SIZE = 2

half = DURATION // 2
SAMPLE_STEP = int(half * SAMPLE_RATE)
STEP = int(half * FRAME_HZ)
# Offset in samples.
# Not strictly safe/deterministic.
# We get gradients up to 312/16000 = 0.0195 seconds -> 20ms into the future
PAD = 312


@pytest.fixture
def batch():
    n_sample = int(SAMPLE_RATE * DURATION)
    n_frames = int(DURATION * FRAME_HZ)
    n_frames_horizon = n_frames + 200
    waveform = torch.randn(BATCH_SIZE, n_sample)
    vad = torch.randint(0, 2, (BATCH_SIZE, n_frames_horizon, 2), dtype=torch.float)
    vad_history = torch.rand((BATCH_SIZE, n_frames, 5))
    return {"waveform": waveform, "vad": vad, "vad_history": vad_history}


@pytest.mark.cpc
@pytest.mark.models
@pytest.mark.causality
@pytest.mark.parametrize(
    ("output_layer", "ulm_layers"), [(1, 0), (0, 0), (1, 1), (0, 1)]
)
def test_causality(batch, output_layer, ulm_layers):
    conf = VPModel.load_config()
    conf["encoder"]["output_layer"] = output_layer
    conf["ulm"]["num_layers"] = ulm_layers
    model = VPModel(conf)

    if torch.cuda.is_available():
        model.to("cuda")

    batch = to_device(batch, model.device)
    ##############################################

    batch["waveform"].requires_grad = True
    loss, _, batch = model.shared_step(batch, reduction="none")
    loss = loss["vp"]
    # backward
    loss[:, STEP].norm().backward()
    g = batch["waveform"].grad.abs()

    assert (
        g[:, SAMPLE_STEP + PAD :].sum() == 0
    ), f"Non-Zero gradient after STEP. EncOut: {output_layer}, ULM-layers: {ulm_layers}"
