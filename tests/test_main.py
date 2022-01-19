import pytest
from os.path import join
from os import cpu_count

import torch
import pytorch_lightning as pl

from datasets_turntaking import DialogAudioDM
from conv_ssl.models import ProjectionMetricCallback
from conv_ssl.ulm_projection import ULMProjection
from conv_ssl.utils import repo_root


@pytest.fixture
def dm():
    data_conf = DialogAudioDM.load_config()
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        vad_hz=data_conf["dataset"]["vad_hz"],
        vad_bin_times=data_conf["dataset"]["vad_bin_times"],
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=10,
        num_workers=cpu_count(),
    )
    dm.prepare_data()
    return dm


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
