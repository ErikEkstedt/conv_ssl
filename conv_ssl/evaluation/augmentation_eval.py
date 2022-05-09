from os.path import join
from os import makedirs
from omegaconf import DictConfig, OmegaConf
import hydra

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from conv_ssl.callbacks import SymmetricSpeakersCallback, FlattenPitchCallback
from conv_ssl.model import VPModel
from conv_ssl.utils import (
    everything_deterministic,
    write_json,
    read_json,
    tensor_dict_to_json,
)
from datasets_turntaking import DialogAudioDM

everything_deterministic()

"""

savepath/checkpoint_path/threshold_path must be full-paths or relative to `hydra-directory`

python conv_ssl/evaluation/augmentation_eval.py \
        +checkpoint_path=/path/to/chkpt \
        +savepath=/save/path \
        +threshold_path=/path/threshholds.json \

additional flags:
    +flat_statistic=mean  # mean/median default=mean
    +target_f0=200  # default=-1, only used if all F0 should take the same value
"""


def test_flat(
    model, dloader, flat_params, max_batches=None, project="VAPFlatTest", online=False
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

    # Limit batches
    if max_batches is not None:
        trainer = Trainer(
            gpus=-1,
            limit_test_batches=max_batches,
            deterministic=True,
            logger=logger,
            callbacks=[
                FlattenPitchCallback(**flat_params),
                SymmetricSpeakersCallback(),
            ],
        )
    else:
        trainer = Trainer(
            gpus=-1,
            deterministic=True,
            logger=logger,
            callbacks=[
                FlattenPitchCallback(**flat_params),
                SymmetricSpeakersCallback(),
            ],
        )

    result = trainer.test(model, dataloaders=dloader, verbose=False)
    return result


@hydra.main(config_path="../conf", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate model"""
    cfg_dict = OmegaConf.to_object(cfg)
    cfg_dict = dict(cfg_dict)

    # Save directory
    savepath = cfg.get("savepath", None)
    assert savepath is not None, f"Please provide savepath by `+savepath=/save/path"
    savepath = savepath + "_flat"

    threshold_path = cfg.get("threshold_path", None)
    assert (
        threshold_path is not None
    ), f"Please provide thresholds by `+thresholds=/thresh/path"

    ##################################
    # Load Model
    ##################################
    model = VPModel.load_from_checkpoint(cfg.checkpoint_path, strict=False)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    ##################################
    # Flat params
    ##################################
    flat_params = {
        "target_f0": cfg.get("target_f0", -1),
        "statistic": cfg.get("statistic", "mean"),
        "stats_frame_length": int(0.05 * model.sample_rate),
        "stats_hop_length": int(0.02 * model.sample_rate),
        "sample_rate": model.sample_rate,
        "to_mono": True,
    }
    for k, v in flat_params.items():
        print(f"{k}: {v}")
    print("#" * 60)

    ##################################
    # Load Data
    ##################################
    data_conf = model.conf["data"]
    data_conf["audio_mono"] = False
    data_conf["datasets"] = cfg_dict["data"].get("datasets", data_conf["datasets"])
    data_conf["batch_size"] = cfg_dict["data"].get(
        "batch_size", data_conf["batch_size"]
    )
    data_conf["num_workers"] = cfg_dict["data"].get(
        "num_workers", data_conf["num_workers"]
    )
    print("Num Workers: ", data_conf["num_workers"])
    print("Batch size: ", data_conf["batch_size"])
    print("Mono: ", data_conf["audio_mono"])
    print("datasets: ", data_conf["datasets"])
    input("Press Enter to Continue")
    dm = DialogAudioDM(**data_conf)
    dm.prepare_data()
    dm.setup("test")

    ##################################
    # Thresholds
    ##################################
    print("Loading thresholds: ", threshold_path)
    thresholds = read_json(threshold_path)

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
    result = test_flat(
        model, dm.test_dataloader(), flat_params=flat_params, online=False
    )[0]
    metrics = model.test_metric.compute()

    metrics["loss"] = result["test_loss"]
    metrics["threshold_pred_shift"] = thresholds["pred_shift"]
    metrics["threshold_pred_bc"] = thresholds["pred_bc"]
    metrics["threshold_short_long"] = thresholds["short_long"]

    makedirs(savepath, exist_ok=True)
    torch.save(metrics, join(savepath, "metric.pt"))
    metric_json = tensor_dict_to_json(metrics)
    write_json(metric_json, join(savepath, "metric.json"))
    print("Saved metrics -> ", join(savepath, "metric.pt"))


if __name__ == "__main__":
    evaluate()
