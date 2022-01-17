import pytest
from os.path import join
from datasets_turntaking.dm_dialog_audio import DialogAudioDM
from conv_ssl.ulm_projection import ULMProjection
from conv_ssl.utils import repo_root


@pytest.fixture
def batch():
    data_conf = DialogAudioDM.load_config()
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        audio_include_ratio=data_conf["dataset"]["audio_include_ratio"],
        audio_context_duration=data_conf["dataset"]["audio_context_duration"],
        ipu_min_time=data_conf["dataset"]["ipu_min_time"],
        ipu_pause_time=data_conf["dataset"]["ipu_pause_time"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hop_time=data_conf["dataset"]["vad_hop_time"],
        vad_bin_sizes=data_conf["dataset"]["vad_bin_sizes"],
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=4,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup()
    return next(iter(dm.val_dataloader()))


@pytest.mark.models
@pytest.mark.ulm_projection
@pytest.mark.parametrize(
    ("quantizer_n_codes", "tier1_num_layer", "tier2_num_layer"),
    [(0, 0, 1), (0, 0, 2), (100, 1, 1), (100, 0, 1), (100, 1, 0)],
)
def test_ulm_projection(batch, quantizer_n_codes, tier1_num_layer, tier2_num_layer):
    conf = ULMProjection.load_config()
    conf["quantizer"]["n_codes"] = quantizer_n_codes
    conf["tier1"]["num_layers"] = tier1_num_layer
    conf["tier2"]["num_layers"] = tier2_num_layer
    model = ULMProjection(conf)
    _ = model.shared_step(batch)


@pytest.mark.models
@pytest.mark.ulm_projection
@pytest.mark.encoder
@pytest.mark.parametrize(
    "name", ["wav2vec", "vq_wav2vec", "wavlm_base+", "hubert_base"]
)
def test_ulm_projection_encoders(batch, name):
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
    _ = model.shared_step(batch)


@pytest.mark.models
@pytest.mark.ulm_projection
@pytest.mark.omit_discrete
@pytest.mark.parametrize(
    ("tier2_num_layer", "tier2_dim"),
    [(1, 768), (1, 64), (2, 64)],
)
def test_ulm_projection_non_discrete(batch, tier2_num_layer, tier2_dim):
    """
    Try tier2 dimensions different from encoder-output
    """
    conf = ULMProjection.load_config()
    conf["quantizer"]["n_codes"] = 0
    conf["tier1"]["num_layers"] = 0
    conf["tier2"]["num_layers"] = tier2_num_layer
    conf["tier2"]["dim"] = tier2_dim
    model = ULMProjection(conf)
    _ = model.shared_step(batch)


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


@pytest.mark.models
@pytest.mark.ulm_projection
@pytest.mark.head
@pytest.mark.parametrize("regression", [False, True])
def test_tier2(batch, regression):
    """Test vad-prediction head regression"""
    conf = ULMProjection.load_config()
    conf["vad_class_prediction"]["regression"] = regression
    model = ULMProjection(conf)
    _ = model.shared_step(batch)
