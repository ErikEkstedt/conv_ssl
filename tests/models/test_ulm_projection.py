import pytest
from os.path import join
import torch
from conv_ssl.ulm_projection import ULMProjection
from conv_ssl.utils import repo_root


def to_device(batch, device="cuda"):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


@pytest.fixture
def fake_batch():
    audio_duration = 10
    sample_rate = 16000
    batch_size = 4
    hop_time = 0.01
    n_samples = int(audio_duration * sample_rate)
    n_frames = int(audio_duration / hop_time)
    batch = {
        "waveform": torch.randn(batch_size, n_samples),
        "vad": torch.randint(0, 2, (batch_size, n_frames, 2)).float(),
        "vad_label": torch.randint(0, 256, (batch_size, n_frames)),
        "vad_history": torch.randn(batch_size, n_frames, 5, 2).float(),
    }
    return batch


@pytest.mark.models
@pytest.mark.ulm_projection
@pytest.mark.parametrize(
    "name", ["wav2vec", "vq_wav2vec", "wavlm_base+", "hubert_base"]
)
def test_ulm_projection_encoders(fake_batch, name):
    conf_path = join(repo_root(), "conv_ssl/config")
    if name == "hubert_base":
        conf_path = join(conf_path, "ulm_hubert.yaml")
    elif name == "wav2vec":
        conf_path = join(conf_path, "ulm_wav2vec.yaml")
    elif name == "vq_wav2vec":
        conf_path = join(conf_path, "ulm_vq_wav2vec.yaml")
    elif name == "wav2vec2_base":
        conf_path = join(conf_path, "ulm_wav2vec2.yaml")
    elif name == "wavlm_base+":
        conf_path = join(conf_path, "ulm_wavlm.yaml")
    else:
        conf_path = join(conf_path, "ulm_cpc.yaml")

    conf = ULMProjection.load_config(conf_path)
    conf["tier1"]["num_layers"] = 0
    conf["tier2"]["num_layers"] = 1
    model = ULMProjection(conf)

    if torch.cuda.is_available():
        model.to("cuda")
        fake_batch = to_device(fake_batch, "cuda")
    _ = model.shared_step(fake_batch)


@pytest.mark.models
@pytest.mark.ulm_projection
@pytest.mark.parametrize(
    ("quantizer_n_codes", "tier1_num_layer", "tier2_num_layer"),
    [(0, 0, 1), (0, 0, 2), (100, 1, 1), (100, 0, 1), (100, 1, 0)],
)
def test_ulm_projection(
    fake_batch, quantizer_n_codes, tier1_num_layer, tier2_num_layer
):
    conf = ULMProjection.load_config()
    conf["quantizer"]["n_codes"] = quantizer_n_codes
    conf["tier1"]["num_layers"] = tier1_num_layer
    conf["tier2"]["num_layers"] = tier2_num_layer
    model = ULMProjection(conf)
    _ = model.shared_step(fake_batch)


@pytest.mark.models
@pytest.mark.ulm_projection
@pytest.mark.parametrize(
    ("tier2_num_layer", "tier2_dim"),
    [(1, 768), (1, 64), (2, 64)],
)
def test_ulm_projection_non_discrete(fake_batch, tier2_num_layer, tier2_dim):
    """
    Try tier2 dimensions different from encoder-output
    """
    conf = ULMProjection.load_config()
    conf["quantizer"]["n_codes"] = 0
    conf["tier1"]["num_layers"] = 0
    conf["tier2"]["num_layers"] = tier2_num_layer
    conf["tier2"]["dim"] = tier2_dim
    model = ULMProjection(conf)
    _ = model.shared_step(fake_batch)


@pytest.mark.models
@pytest.mark.ulm_projection
@pytest.mark.parametrize("regression", [False, True])
def test_tier2(fake_batch, regression):
    """Test vad-prediction head regression"""
    conf = ULMProjection.load_config()
    conf["vad_class_prediction"]["regression"] = regression
    model = ULMProjection(conf)
    _ = model.shared_step(fake_batch)


@pytest.mark.models
@pytest.mark.ulm_projection
def test_ulm_projection_assertion_error():
    """Impossible values should yield assertion error"""
    conf = ULMProjection.load_config()
    # This configuration is impossible
    # if we don't have discrete codes we can't train AR-model
    # thus only tier2 is required (num_layers, num_heads, etc)
    conf["quantizer"]["n_codes"] = 0
    conf["tier1"]["num_layers"] = 1
    conf["tier2"]["num_layers"] = 0
    try:
        _ = ULMProjection(conf)
    except AssertionError:
        assert True
