from os.path import join, split, basename
from argparse import ArgumentParser
from os import makedirs, environ, cpu_count

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from datamodule_disk import DiskDM
from conv_ssl.model import VPModel
from conv_ssl.utils import count_parameters, everything_deterministic

import wandb

PROJECT = "VPModel"
SAVEDIR = "runs/VPModel"


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


def train():
    parser = ArgumentParser()
    parser = VPModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name_info", type=str, default="")
    parser.add_argument("--project_info", type=str, default="")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--log_gradients", action="store_true")
    parser.add_argument("--animation", action="store_true")
    parser.add_argument("--animation_epoch_start", default=10, type=int)
    parser.add_argument("--animation_n", default=10, type=int)
    parser.add_argument("--data_root", default="swb_dataset", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=cpu_count(), type=int)
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    args.deterministic = True

    local_rank = environ.get("LOCAL_RANK", 0)

    #########
    # Model #
    #########
    conf = VPModel.load_config(path=args.conf, args=args)
    model = VPModel(conf)

    # print after callbacks/wandb init
    if local_rank == 0:
        print("-" * 60)
        print(model.summary())
        print(f"Model Name: {model.run_name}")
        print("Base: ", args.conf)
        print("PARAMETERS: ", count_parameters(model))
        print()
        print("-" * 60)

    dm = DiskDM(
        args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Callbacks & Logger
    logger = None
    callbacks = []

    # this should be handled automatically with pytorch_lightning?
    if not args.fast_dev_run:
        makedirs(SAVEDIR, exist_ok=True)
        logger = WandbLogger(
            save_dir=SAVEDIR,
            project=PROJECT + args.project_info,
            name=model.run_name + args.name_info,
            log_model=not args.dont_log_model,
            # log_model=True,  # True: logs after training finish
        )

        callbacks.append(
            ModelCheckpoint(
                mode="max",
                monitor="val_f1_weighted",
                # mode="min",
                # monitor="val_loss",
            )
        )
        callbacks.append(WandbArtifactCallback())
        verbose = False
        if local_rank == 0:
            print(f"Early stopping (patience={args.patience})")
            verbose = True

        callbacks.append(
            EarlyStopping(
                monitor="val_f1_weighted",
                mode="max",
                patience=args.patience,
                strict=True,  # crash if "monitor" is not found in val metrics
                verbose=verbose,
            )
        )

    # Trainer
    # args.auto_lr_find = True
    trainer = pl.Trainer.from_argparse_args(
        args=args, logger=logger, callbacks=callbacks
    )
    # auto_finder = trainer.tune(model, dm)["lr_find"]

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train()