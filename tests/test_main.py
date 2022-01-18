import pytest
from os.path import join
from os import cpu_count

import torch
import pytorch_lightning as pl

from datasets_turntaking.dm_dialog_audio import DialogAudioDM
from conv_ssl.models import ProjectionMetricCallback
from conv_ssl.ulm_projection import ULMProjection
from conv_ssl.utils import repo_root


@pytest.fixture
def dm():
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
        batch_size=cpu_count(),
    )
    dm.prepare_data()
    return dm


# @pytest.mark.main
# @pytest.mark.parametrize(
#     ("quantizer_n_codes", "tier1_num_layer", "tier2_num_layer"),
#     [(0, 0, 1), (0, 0, 2), (100, 1, 1), (100, 0, 1), (100, 1, 0)],
# )
# def test_ulm_projection(batch, quantizer_n_codes, tier1_num_layer, tier2_num_layer):
#     conf = ULMProjection.load_config()
#     conf["quantizer"]["n_codes"] = quantizer_n_codes
#     conf["tier1"]["num_layers"] = tier1_num_layer
#     conf["tier2"]["num_layers"] = tier2_num_layer
#     model = ULMProjection(conf)
#     _ = model.shared_step(batch)


@pytest.mark.main
@pytest.mark.parametrize(
    "name", ["wav2vec", "vq_wav2vec", "wavlm_base+", "hubert_base"]
)
def test_ulm_projection_encoders(dm, name):
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

    logger = None
    callbacks = [ProjectionMetricCallback()]

    trainer = pl.Trainer(gpus=-1, fast_dev_run=1, logger=logger, callbacks=callbacks)
    # auto_finder = trainer.tune(model, dm)["lr_find"]
    trainer.fit(model, datamodule=dm)
