import pytest

import torch
from conv_ssl.model import VPModel
from conv_ssl.utils import everything_deterministic, to_device, load_hydra_conf

everything_deterministic()

BATCH_SIZE = 2
DURATION = 10
# Offset in samples.
# Not strictly safe/deterministic.
# We get gradients up to 312/16000 = 0.0195 seconds -> 20ms into the future
PAD = 312


def get_sample_batch(sample_rate, frame_hz, frame_horizon):
    n_sample = int(sample_rate * DURATION)
    n_frames = int(DURATION * frame_hz)
    n_frames_horizon = n_frames + frame_horizon
    waveform = torch.randn(BATCH_SIZE, n_sample)
    vad = torch.randint(0, 2, (BATCH_SIZE, n_frames_horizon, 2), dtype=torch.float)
    vad_history = torch.rand((BATCH_SIZE, n_frames, 5))
    return {"waveform": waveform, "vad": vad, "vad_history": vad_history}


@pytest.mark.causality
@pytest.mark.parametrize(
    ("output_layer", "config_name"),
    [
        (0, "model/discrete"),
        (1, "model/discrete"),
        (0, "model/discrete_20hz"),
        (1, "model/discrete_20hz"),
        (0, "model/discrete_50hz"),
        (1, "model/discrete_50hz"),
    ],
)
def test_causality(output_layer, config_name):

    conf = load_hydra_conf()
    conf["model"] = load_hydra_conf(config_name=config_name)["model"]
    conf["model"]["encoder"]["output_layer"] = output_layer
    model = VPModel(conf)

    if torch.cuda.is_available():
        model.to("cuda")

    sample_rate = conf["model"]["encoder"]["sample_rate"]
    frame_hz = conf["model"]["encoder"]["frame_hz"]
    frame_horizon = model.VAP.horizon_frames

    batch = get_sample_batch(
        sample_rate=sample_rate, frame_hz=frame_hz, frame_horizon=frame_horizon
    )
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")
    n_frames = batch["vad_history"].shape[1]
    half_frames = n_frames // 2
    half_samples = batch["waveform"].shape[-1] // 2

    batch = to_device(batch, model.device)
    ##############################################
    batch["waveform"].requires_grad = True
    loss, _, batch = model.shared_step(batch, reduction="none")
    l = loss["frames"]
    # backward
    l[:, half_frames].sum().backward()
    g = batch["waveform"].grad.abs()
    # g[:, half_samples + PAD :].sum()

    assert (
        g[:, half_samples + PAD :].sum() == 0
    ), f"Non-Zero gradient after STEP. EncOut: {output_layer}, ULM-layers: {ulm_layers}"
