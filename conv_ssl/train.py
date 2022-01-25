from argparse import ArgumentParser
from os import makedirs, environ
from os.path import join, split, basename


import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from datasets_turntaking import DialogAudioDM
from conv_ssl.models import ProjectionMetricCallback
from conv_ssl.ulm_projection import ULMProjection
from conv_ssl.utils import count_parameters


# import wandb


PROJECT = "ULMProjection"
SAVEDIR = "runs/conv_ssl/ULMProjection"


def train():
    parser = ArgumentParser()
    parser = ULMProjection.add_model_specific_args(parser)
    parser = DialogAudioDM.add_data_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name_info", type=str, default="")
    parser.add_argument("--project_info", type=str, default="")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--log_gradients", action="store_true")
    parser.add_argument("--animation", action="store_true")
    parser.add_argument("--animation_epoch_start", default=10, type=int)
    parser.add_argument("--animation_n", default=10, type=int)
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    local_rank = environ.get("LOCAL_RANK", 0)

    #########
    # Model #
    #########
    conf = ULMProjection.load_config(path=args.conf, args=args)
    model = ULMProjection(conf)

    ##############
    # DataModule #
    ##############
    data_conf = DialogAudioDM.load_config(path=args.data_conf, args=args)
    data_conf["dataset"]["vad_hz"] = model.encoder.frame_hz
    if local_rank == 0:
        DialogAudioDM.print_dm(data_conf, args)

    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hz=data_conf["dataset"]["vad_hz"],
        vad_bin_times=data_conf["dataset"]["vad_bin_times"],
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.prepare_data()

    # Callbacks & Logger
    logger = None
    callbacks = [
        ProjectionMetricCallback(
            regression=conf["vad_class_prediction"]["regression"],
            bin_sizes=conf["vad_class_prediction"]["bin_sizes"],
        )
    ]

    # this should be handled automatically with pytorch_lightning?
    if not args.fast_dev_run:
        # logger, callbacks = add_standard_callbacks(name, args, model, callbacks)
        # if args.animation:
        #     callbacks = add_animator_callback(args, callbacks)
        makedirs(SAVEDIR, exist_ok=True)
        logger = WandbLogger(
            save_dir=SAVEDIR,
            project=PROJECT + args.project_info,
            name=model.run_name + args.name_info,
            log_model=not args.dont_log_model,
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="{epoch}-{val_loss:.5f}",
                save_top_k=1,
                # mode="min",
                # monitor="val_loss",
                monitor="val/f1_weighted",
                mode="max",
            )
        )
        verbose = False
        if local_rank == 0:
            print(f"Early stopping (patience={args.patience})")
            verbose = True
        callbacks.append(
            EarlyStopping(
                # monitor="val_loss_vp",
                monitor="val/f1_weighted",
                mode="max",
                patience=args.patience,
                strict=True,  # crash if "monitor" is not found in val metrics
                verbose=verbose,
            )
        )

    # print after callbacks/wandb init
    if local_rank == 0:
        print("-" * 60)
        print(model.summary())
        print(f"Model Name: {model.run_name}")
        print("Base: ", args.conf)
        print("PARAMETERS: ", count_parameters(model))
        print()
        print("-" * 60)

    # Trainer
    # args.auto_lr_find = True
    trainer = pl.Trainer.from_argparse_args(
        args=args, logger=logger, callbacks=callbacks
    )
    # auto_finder = trainer.tune(model, dm)["lr_find"]
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train()
