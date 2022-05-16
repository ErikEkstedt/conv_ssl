from os.path import join, basename
from pathlib import Path
from os import makedirs
from omegaconf import DictConfig, OmegaConf
import hydra

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from conv_ssl.callbacks import (
    SymmetricSpeakersCallback,
    FlattenPitchCallback,
    ShiftPitchCallback,
    LowPassFilterCallback,
    NeutralIntensityCallback,
)
from conv_ssl.model import VPModel
from conv_ssl.utils import (
    everything_deterministic,
    write_json,
    read_json,
    tensor_dict_to_json,
)
from datasets_turntaking import DialogAudioDM

everything_deterministic()

SAVEPATH = "/home/erik/projects/CCConv/conv_ssl/assets/PaperB/eval"


def load_dm(model, cfg_dict, verbose=False):
    data_conf = model.conf["data"]
    data_conf["audio_mono"] = False
    data_conf["datasets"] = cfg_dict["data"].get("datasets", data_conf["datasets"])
    data_conf["batch_size"] = cfg_dict["data"].get(
        "batch_size", data_conf["batch_size"]
    )
    data_conf["num_workers"] = cfg_dict["data"].get(
        "num_workers", data_conf["num_workers"]
    )
    if verbose:
        print("datasets: ", data_conf["datasets"])
        print("duration: ", data_conf["audio_duration"])
        print("Num Workers: ", data_conf["num_workers"])
        print("Batch size: ", data_conf["batch_size"])
        print("Mono: ", data_conf["audio_mono"])
    dm = DialogAudioDM(**data_conf)
    dm.prepare_data()
    dm.setup(None)
    return dm


def get_augmentation_params(model, cfg, augmentation):
    name_suffix = ""
    aug_params = {}
    if augmentation == "flat_f0":
        aug_params = {
            "target_f0": cfg.get("target_f0", -1),
            "statistic": cfg.get("statistic", "mean"),
            "stats_frame_length": int(0.05 * model.sample_rate),
            "stats_hop_length": int(0.02 * model.sample_rate),
            "sample_rate": model.sample_rate,
            "to_mono": True,
        }
        name_suffix = "flat_f0"
    elif augmentation == "shift_f0":
        aug_params = {
            "factor": cfg.get("factor", 0.9),
            "sample_rate": model.sample_rate,
            "to_mono": True,
        }
        name_suffix = f"shift_f0_{aug_params['factor']}"
    elif augmentation == "flat_intensity":
        aug_params = {
            "vad_hz": model.frame_hz,
            "vad_cutoff": cfg.get("vad_cutoff", 0.2),
            "hop_time": cfg.get("hop_time", 0.01),
            "f0_min": cfg.get("f0_min", 60),
            "statistic": cfg.get("statistic", "mean"),
            "sample_rate": model.sample_rate,
            "to_mono": True,
        }
        name_suffix = "flat_intensity"
    elif augmentation == "only_f0":
        aug_params = {
            "cutoff_freq": cfg.get("cutoff_freq", 400),
            "sample_rate": model.sample_rate,
            "norm": True,
            "to_mono": True,
        }
        name_suffix = f"_only_f0_{aug_params['cutoff_freq']}"
    return aug_params, name_suffix


def test_augmented(
    model,
    dloader,
    augmentation,
    aug_params,
    max_batches=None,
    project="VAPFlatTest",
    online=False,
):
    """
    Iterate over the dataloader to extract metrics.

    Callbacks are done in order! so important to do Flat first...

    * Adds FlattenPitchCallback
        - flattens the pitch of each waveform/speaker
    * Adds SymmetricSpeakersCallback
        - each sample is duplicated with channels reversed
    * online = True
        - upload to wandb
    """
    logger = None
    if online:
        logger = WandbLogger(
            project=project,
            name=model.run_name,
            log_model=False,
        )

    callbacks = []

    if augmentation == "flat_f0":
        callbacks.append(FlattenPitchCallback(**aug_params))
    elif augmentation == "shift_f0":
        callbacks.append(ShiftPitchCallback(**aug_params))
    elif augmentation == "flat_intensity":
        callbacks.append(NeutralIntensityCallback(**aug_params))
    elif augmentation == "only_f0":
        callbacks.append(LowPassFilterCallback(**aug_params))
    else:
        raise NotImplementedError(
            f"{augmentation} not implemented. Choose ['only_f0', 'flat_pitch', 'shift_f0', 'flat_intensity']"
        )
    callbacks.append(SymmetricSpeakersCallback())

    # Limit batches
    if max_batches is not None:
        trainer = Trainer(
            gpus=-1,
            limit_test_batches=max_batches,
            deterministic=True,
            logger=logger,
            callbacks=callbacks,
        )
    else:
        trainer = Trainer(
            gpus=-1, deterministic=True, logger=logger, callbacks=callbacks
        )

    result = trainer.test(model, dataloaders=dloader, verbose=False)
    return result


@hydra.main(config_path="../conf", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate model"""
    cfg_dict = OmegaConf.to_object(cfg)
    cfg_dict = dict(cfg_dict)

    assert cfg.get("checkpoint_path", False), "Must provide `+checkpoint_path=/path/to`"

    augmentation = cfg.get("augmentation", None)
    assert (
        augmentation is not None
    ), f"Please provide `augmentation` by `+augmentation=` and any of ['flat_f0', 'shift_f0', 'flat_intensity', 'only_f0']"

    ##################################
    # Load Model
    ##################################
    model = VPModel.load_from_checkpoint(cfg.checkpoint_path, strict=False)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    # Save directory + Augmentations
    savepath = join(SAVEPATH, basename(cfg.checkpoint_path).replace(".ckpt", ""))
    savepath += "_" + "_".join(cfg.data.datasets)
    Path(savepath).mkdir(exist_ok=True, parents=True)
    aug_params, name_suffix = get_augmentation_params(model, cfg, augmentation)

    for k, v in aug_params.items():
        print(f"{k}: {v}")
    print("#" * 60)

    # ##################################
    # # Load Data
    # ##################################
    data_conf = model.conf["data"]
    dm = load_dm(model, cfg_dict, verbose=True)

    print("Thresholds")
    threshold_path = join(savepath, "thresholds.json")
    print("Loading thresholds: ", threshold_path)
    thresholds = read_json(threshold_path)
    for k, v in thresholds.items():
        print(f"{k}: {v}")

    print(f"SAVEPATH: ", savepath)
    # input("Press Enter to Continue")

    ##################################
    # Test
    ##################################
    print("#" * 60)
    print("Flat Score (test-set)...")
    print("#" * 60)
    model.test_metric = model.init_metric(
        threshold_pred_shift=thresholds["pred_shift"],
        threshold_short_long=thresholds["short_long"],
        threshold_bc_pred=thresholds["pred_bc"],
    )
    result = test_augmented(
        model,
        dm.test_dataloader(),
        augmentation=augmentation,
        aug_params=aug_params,
        online=False,
        max_batches=cfg_dict.get("max_batches", None),
    )[0]
    metrics = model.test_metric.compute()
    metrics["loss"] = result["test_loss"]
    metrics["threshold_pred_shift"] = thresholds["pred_shift"]
    metrics["threshold_pred_bc"] = thresholds["pred_bc"]
    metrics["threshold_short_long"] = thresholds["short_long"]

    makedirs(savepath, exist_ok=True)
    metric_json = tensor_dict_to_json(metrics)
    filepath = join(savepath, f"metric_{name_suffix}.json")
    write_json(metric_json, filepath)
    print("Saved metrics -> ", filepath)


if __name__ == "__main__":
    evaluate()
