import pytest
import pytorch_lightning as pl

from conv_ssl.model import VPModel
from conv_ssl.utils import everything_deterministic
from datasets_turntaking import DialogAudioDM

everything_deterministic()


@pytest.fixture
def dm_100hz():
    data_conf = DialogAudioDM.load_config()
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hz=100,
        vad_bin_times=data_conf["dataset"]["vad_bin_times"],
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=4,
        num_workers=0,
    )
    dm.prepare_data()
    return dm


@pytest.mark.main
@pytest.mark.parametrize(
    ("name", "output_layer"),
    [
        ("cpc", 0),
        ("cpc", 1),
        ("wav2vec", 0),
        ("wav2vec", 1),
        ("vq_wav2vec", 0),
        ("vq_wav2vec", 1),
    ],
)
def test_train(dm_100hz, name, output_layer):
    conf = VPModel.load_config()
    conf["encoder"]["type"] = name
    conf["encoder"]["output_layer"] = output_layer
    conf["quantizer"]["n_codes"] = 0
    conf["quantizer"]["vector_path"] = None
    conf["ulm"]["num_layers"] = 0
    conf["ar"]["num_layers"] = 1
    model = VPModel(conf)

    trainer = pl.Trainer(gpus=-1, fast_dev_run=1, strategy="ddp", deterministic=True)
    trainer.fit(model, datamodule=dm_100hz)
