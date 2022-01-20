import pytest
from os import cpu_count
import pytorch_lightning as pl

from datasets_turntaking import DialogAudioDM
from conv_ssl.models import ProjectionMetricCallback
from conv_ssl.ulm_projection import ULMProjection


@pytest.fixture
def dm_50hz():
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


@pytest.fixture
def dm_100hz():
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
    ("name", "output_layer"),
    [
        ("wavlm_base+", 0),
        ("wavlm_base+", 6),
        ("hubert_base", 0),
        ("hubert_base", 6),
    ],
)
def test_ulm_projection_50hz(dm_50hz, name, output_layer):
    conf = ULMProjection.load_config()
    conf["encoder"]["type"] = name
    conf["encoder"]["output_layer"] = output_layer
    conf["quantizer"]["n_codes"] = 0
    conf["tier1"]["num_layers"] = 0
    conf["tier2"]["num_layers"] = 1
    conf["tier2"]["dim"] = 256
    model = ULMProjection(conf)

    logger = None
    callbacks = [ProjectionMetricCallback()]
    trainer = pl.Trainer(
        gpus=-1, fast_dev_run=1, strategy="ddp", logger=logger, callbacks=callbacks
    )
    trainer.fit(model, datamodule=dm_50hz)


@pytest.mark.main
@pytest.mark.parametrize(
    ("name", "output_layer"),
    [
        ("wav2vec", 0),
        ("wav2vec", 6),
        ("vq_wav2vec", 0),
        ("vq_wav2vec", 6),
        ("cpc", 0),
        ("cpc", 6),
    ],
)
def test_ulm_projection_100hz(dm_100hz, name, output_layer):
    conf = ULMProjection.load_config()
    conf["encoder"]["type"] = name
    conf["encoder"]["output_layer"] = output_layer
    conf["quantizer"]["n_codes"] = 0
    conf["tier1"]["num_layers"] = 0
    conf["tier2"]["num_layers"] = 1
    conf["tier2"]["dim"] = 256
    model = ULMProjection(conf)

    logger = None
    callbacks = [ProjectionMetricCallback()]
    trainer = pl.Trainer(
        gpus=-1, fast_dev_run=1, strategy="ddp", logger=logger, callbacks=callbacks
    )
    trainer.fit(model, datamodule=dm_100hz)
