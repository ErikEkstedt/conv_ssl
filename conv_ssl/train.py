from argparse import ArgumentParser
from os import makedirs, environ
from os.path import join, split, basename


import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from conv_ssl.models import ProjectionMetricCallback
from conv_ssl.ulm_projection import ULMProjection
from conv_ssl.utils import count_parameters
from datasets_turntaking.dm_dialog_audio import (
    DialogAudioDM,
    DialogIPU,
    print_dm,
    get_dialog_audio_datasets,
)

import wandb


PROJECT = "ULMProjection"
SAVEDIR = "runs/conv_ssl/ULMProjection"


class AnimationCallback(pl.Callback):
    def __init__(
        self,
        sample_dset,
        n_ani=-1,
        frame_step=5,
        start_epoch=1,
        cache_path="/tmp/vad_animation",
    ):
        super().__init__()
        self.n_ani = n_ani
        self.sample_dset = sample_dset
        self.start_epoch = start_epoch
        self.cache_path = cache_path
        self.frame_step = frame_step

    def create_animations(self, model):
        paths = []
        for i, d in tqdm(
            enumerate(self.sample_dset), desc="Animation", total=self.n_ani
        ):
            if self.n_ani > 0 and i == self.n_ani:
                break

            path = join(self.cache_path, f"ani_{i}.mp4")
            model.animate_sample(
                input_ids=d["q"],
                waveform=d["waveform"],
                vad=d["vad"],
                frame_step=self.frame_step,
                path=path,
            )
            paths.append(path)
        return paths

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if trainer.current_epoch < self.start_epoch:
            return None

        paths = self.create_animations(model=pl_module)
        for i, p in enumerate(paths):
            wandb.log(
                {
                    f"animation_{i}": wandb.Video(
                        data_or_path=paths[i], fps=10, format="mp4"
                    )
                }
            )
        return None


def add_standard_callbacks(name, args, model, callbacks):
    makedirs(SAVEDIR, exist_ok=True)
    logger = WandbLogger(
        save_dir=SAVEDIR,
        project=PROJECT + args.project_info,
        name=name + args.name_info,
        log_model=not args.dont_log_model,
    )

    local_rank = environ.get("LOCAL_RANK", 0)
    if local_rank == 0:
        if args.log_gradients:
            logger.watch(model)

        print('path: ', logger.experiment.path)
        id_hash = basename(logger.experiment.path)
        print('id hash: ', id_hash)
        ch_path = join(logger.save_dir, logger.name + "_" + id_hash)
        callbacks.append(
            ModelCheckpoint(
                dirpath=ch_path,
                filename="{epoch}-{val_loss:.5f}",
                save_top_k=1,
                # mode="min",
                # monitor="val_loss",
                monitor="val/f1_weighted",
                mode="max",
            )
        )
        print(f"Early stopping (patience={args.patience})")

        callbacks.append(
            EarlyStopping(
                # monitor="val_loss_vp",
                monitor="val/f1_weighted",
                mode="max",
                patience=args.patience,
                strict=True,  # crash if "monitor" is not found in val metrics
                verbose=True,
            )
        )
    return logger, callbacks


def add_animator_callback(args, callbacks):
    data_conf = DialogAudioDM.load_config(path=args.data_conf, args=args)
    val_hf_dataset = get_dialog_audio_datasets(
        datasets=data_conf["dataset"]["datasets"], split="val"
    )
    sample_dset = DialogIPU(
        dataset=val_hf_dataset,
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hop_time=data_conf["dataset"]["vad_hop_time"],
        vad_bin_sizes=data_conf["dataset"]["vad_bin_sizes"],
    )
    callbacks.append(
        AnimationCallback(
            sample_dset,
            start_epoch=args.animation_epoch_start,
            n_ani=args.animation_n,
        )
    )
    return callbacks


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

    ##############
    # DataModule #
    ##############
    data_conf = DialogAudioDM.load_config(path=args.data_conf, args=args)
    if local_rank == 0:
        print_dm(data_conf, args)
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
        shuffle_training_data=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.prepare_data()

    #########
    # Model #
    #########
    conf = ULMProjection.load_config(path=args.conf, args=args)
    model = ULMProjection(conf)
    name = model.run_name  # Name the run e.g. hubert_44_41

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
            name=name + args.name_info,
            log_model=not args.dont_log_model,
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath='checkpoints',
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
        print(f"Model Name: {name}")
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
