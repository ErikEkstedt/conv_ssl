from omegaconf import DictConfig, OmegaConf
from os import makedirs, environ
import hydra
import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from conv_ssl.callbacks import WandbArtifactCallback
from conv_ssl.model import VPModel
from conv_ssl.utils import everything_deterministic
from datasets_turntaking import DialogAudioDM


everything_deterministic()


@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_object(cfg)
    cfg_dict = dict(cfg_dict)

    if "debug" in cfg_dict:
        environ["WANDB_MODE"] = "offline"
        print("DEBUG -> OFFLINE MODE")

    pl.seed_everything(cfg_dict["seed"])
    local_rank = environ.get("LOCAL_RANK", 0)

    model = VPModel(cfg_dict)

    if cfg_dict["verbose"]:
        print("DataModule")
        for k, v in cfg_dict["data"].items():
            print(f"{k}: {v}")
        print("#" * 60)

    dm = DialogAudioDM(**cfg_dict["data"])
    dm.prepare_data()

    if cfg_dict["trainer"]["fast_dev_run"]:
        trainer = pl.Trainer(**cfg_dict["trainer"])
        print(cfg_dict["model"])
        print("-" * 40)
        print(dm)
        trainer.fit(model, datamodule=dm)
    else:
        # Callbacks & Logger
        logger = WandbLogger(
            # save_dir=SA,
            project=cfg_dict["wandb"]["project"],
            name=model.run_name,
            log_model=False,
        )

        if local_rank == 0:
            print("#" * 40)
            print(f"Early stopping (patience={cfg_dict['early_stopping']['patience']})")
            print("#" * 40)

        callbacks = [
            ModelCheckpoint(
                mode=cfg_dict["checkpoint"]["mode"],
                monitor=cfg_dict["checkpoint"]["monitor"],
            ),
            EarlyStopping(
                monitor=cfg_dict["early_stopping"]["monitor"],
                mode=cfg_dict["early_stopping"]["mode"],
                patience=cfg_dict["early_stopping"]["patience"],
                strict=True,  # crash if "monitor" is not found in val metrics
                verbose=False,
            ),
            LearningRateMonitor(),
            WandbArtifactCallback(),
        ]

        if cfg_dict["optimizer"].get("swa_enable", False):
            callbacks.append(
                StochasticWeightAveraging(
                    swa_epoch_start=cfg_dict["optimizer"].get("swa_epoch_start", 5),
                    annealing_epochs=cfg_dict["optimizer"].get(
                        "swa_annealing_epochs", 10
                    ),
                )
            )

        # Find Best Learning Rate
        trainer = pl.Trainer(gpus=-1)
        lr_finder = trainer.tuner.lr_find(model, dm)
        model.learning_rate = lr_finder.suggestion()
        print("#" * 40)
        print("Initial Learning Rate: ", model.learning_rate)
        print("#" * 40)

        # Actual Training
        trainer = pl.Trainer(
            logger=logger,
            callbacks=callbacks,
            strategy=DDPStrategy(find_unused_parameters=True),
            **cfg_dict["trainer"],
        )
        trainer.fit(model, datamodule=dm)


def load():
    # from conv_ssl.evaluation.utils import load_model
    # checkpoint = "runs/TestHydra/223srezy/checkpoints/epoch=4-step=50.ckpt"
    # model = VPModel.load_from_checkpoint(checkpoint)
    run = wandb.init()
    artifact = run.use_artifact("how_so/TestHydra/3hjsv2z8_model:v0", type="model")
    artifact_dir = artifact.download()
    checkpoint = artifact_dir + "/model"
    print("artifact_dir: ", artifact_dir)
    # run_path = "how_so/TestHydra/3hjsv2z8"
    # model = load_model(run_path=run_path)
    #
    # print(model)


if __name__ == "__main__":
    train()
