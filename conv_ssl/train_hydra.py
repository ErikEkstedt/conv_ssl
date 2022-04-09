from omegaconf import DictConfig, OmegaConf
from os import makedirs, environ
import hydra
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from conv_ssl.model import VPModel
from conv_ssl.utils import everything_deterministic
from datasets_turntaking import DialogAudioDM


everything_deterministic()


class WandbArtifactCallback(pl.Callback):
    def upload(self, trainer):
        run = trainer.logger.experiment
        print(f"Ending run: {run.id}")
        artifact = wandb.Artifact(f"{run.id}_model", type="model")
        for path, val_loss in trainer.checkpoint_callback.best_k_models.items():
            print(f"Adding artifact: {path}")
            artifact.add_file(path)
        run.log_artifact(artifact)

    def on_train_end(self, trainer, pl_module):
        print("Training End ---------------- Custom Upload")
        self.upload(trainer)

    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, KeyboardInterrupt):
            print("Keyboard Interruption ------- Custom Upload")
            self.upload(trainer)


@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_object(cfg)
    cfg_dict = dict(cfg_dict)
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

    # Callbacks & Logger
    logger = None
    callbacks = []
    if not cfg_dict["trainer"]["fast_dev_run"]:
        logger = WandbLogger(
            # save_dir=SA,
            project=cfg_dict["wandb"]["project"],
            name=model.run_name,
            log_model=False,
        )

        callbacks.append(
            ModelCheckpoint(
                mode=cfg_dict["checkpoint"]["mode"],
                monitor=cfg_dict["checkpoint"]["monitor"],
            )
        )
        callbacks.append(WandbArtifactCallback())

        verbose = False
        if local_rank == 0:
            print(f"Early stopping (patience={cfg_dict['early_stopping']['patience']})")
            verbose = True

        callbacks.append(
            EarlyStopping(
                monitor=cfg_dict["early_stopping"]["monitor"],
                mode=cfg_dict["early_stopping"]["mode"],
                patience=cfg_dict["early_stopping"]["patience"],
                strict=True,  # crash if "monitor" is not found in val metrics
                verbose=verbose,
            )
        )

    # Trainer
    trainer = pl.Trainer(**cfg_dict["trainer"], logger=logger, callbacks=callbacks)
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
