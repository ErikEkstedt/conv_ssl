import pytest

import torch
from conv_ssl.model import VPModel
from conv_ssl.evaluation import load_dm, to_device


# def dummy_batch(duration=2, sample_rate=16000, frame_hz=100, batch_size=2):
#     n_sample = int(sample_rate * duration)
#     n_frames = int(duration * frame_hz)
#     n_frames_horizon = n_frames + 200
#     waveform = torch.randn(batch_size, n_sample)
#     vad = torch.randint(0, 2, (batch_size, n_frames_horizon, 2), dtype=torch.float)
#     vad_history = torch.rand((batch_size, n_frames, 5))
#     return {"waveform": waveform, "vad": vad, "vad_history": vad_history}


@pytest.mark.models
@pytest.mark.causality
# @pytest.mark.parametrize(("output_layer, ar_layers"), [(0, 0), (0, 1), (1, 0), (1, 1)])
# @pytest.mark.parametrize("output_layer", [1])
# def test_encoder(output_layer):
def test_encoder():

    # torch.autograd.set_detect_anomaly(True)
    conf = VPModel.load_config()
    conf["encoder"]["output_layer"] = 1
    model = VPModel(conf)
    optim = model.configure_optimizers()

    ##############################################
    dm = load_dm(model)
    batch = to_device(next(iter(dm.val_dataloader())), model.device)
    ##############################################

    sample_rate = 16000
    half = dm.audio_duration // 2
    sample_step = int(half * sample_rate)
    step = int(half * model.net.encoder.frame_hz)
    pad = 312  # not strictly safe. we may see up to 0.0195 seconds -> 20ms into the future

    optim.zero_grad()

    batch["waveform"].requires_grad = True
    loss, _, batch = model.shared_step(batch, reduction="none")
    loss = loss["vp"]
    # backward
    loss[:, step].norm().backward()
    g = batch["waveform"].grad.abs()

    assert (
        g[:, sample_step + pad :].sum() == 0
    ), f"Non-Zero gradient after step. AR:{ar_layer}, EncOut:{output_layer}"
