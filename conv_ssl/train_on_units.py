from argparse import ArgumentParser
import pytorch_lightning as pl

from conv_ssl.dm_units import SegmentDM, SegmentDataset
from conv_ssl.models import ProjectionMetricCallback
from conv_ssl.train import AnimationCallback, add_standard_callbacks
from conv_ssl.ulm_projection import ULMProjection
from conv_ssl.utils import count_parameters

import wandb


def add_animator_callback(k, args, callbacks):
    """Small (validation subset) dataset used for animations"""
    sample_dset = SegmentDataset(root=args.data_root, split="val_audio", k=k)
    callbacks.append(
        AnimationCallback(
            sample_dset,
            start_epoch=args.animation_epoch_start,
            n_ani=args.animation_n,
        )
    )
    return callbacks


def train_on_units():
    parser = ArgumentParser()
    parser = ULMProjection.add_model_specific_args(parser)
    parser = SegmentDM.add_data_specific_args(parser)
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

    ##############
    # DataModule #
    ##############
    print("-" * 60)
    print("Dataloader")
    print("\tdata_root: ", args.data_root)
    print("\tbatch_size: ", args.batch_size)
    print("\tnum_workers: ", args.num_workers)
    print()
    dm = SegmentDM(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    #########
    # Model #
    #########
    conf = ULMProjection.load_config(path=args.conf, args=args)
    model = ULMProjection(conf)
    name = model.run_name  # Name the run e.g. hubert_44_41

    # Callbacks & Logger
    logger = None
    callbacks = []
    callbacks.append(ProjectionMetricCallback())

    # this should be handled automatically with pytorch_lightning?
    if not args.fast_dev_run:
        logger, callbacks = add_standard_callbacks(name, args, model, callbacks)
        if not model.conf["vad_class_prediction"]["regression"]:
            if args.animation:
                callbacks = add_animator_callback(
                    k=conf["quantizer"]["n_codes"], args=args, callbacks=callbacks
                )

    # print after callbacks/wandb init
    print("-" * 60)
    print(model.summary())
    print(f"Model Name: {name}")
    print("Base: ", args.conf)
    print("PARAMETERS: ", count_parameters(model))
    print()

    # Trainer
    # args.auto_lr_find = True
    print("-" * 60)
    trainer = pl.Trainer.from_argparse_args(
        args=args, logger=logger, callbacks=callbacks
    )
    # auto_finder = trainer.tune(model, dm)["lr_find"]

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train_on_units()
