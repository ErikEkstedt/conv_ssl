import pytest
import pytorch_lightning as pl

from conv_ssl.model import VPModel
from conv_ssl.utils import everything_deterministic, load_hydra_conf
from datasets_turntaking import DialogAudioDM

everything_deterministic()


@pytest.mark.main
@pytest.mark.parametrize(
    "config_name",
    [
        "model/discrete",
        "model/discrete_20hz",
        "model/discrete_50hz",
    ],
)
def test_cpc_train(config_name):
    conf = load_hydra_conf()
    conf["model"] = load_hydra_conf(config_name=config_name)["model"]
    model = VPModel(conf)

    conf["data"]["num_workers"] = 0
    conf["data"]["batch_size"] = 4
    dm = DialogAudioDM(**conf["data"])
    dm.prepare_data()

    trainer = pl.Trainer(
        gpus=-1, fast_dev_run=1, strategy="ddp", deterministic=True, log_every_n_steps=1
    )
    trainer.fit(model, datamodule=dm)
