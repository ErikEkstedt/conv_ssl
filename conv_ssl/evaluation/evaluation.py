from pathlib import Path
from os.path import join, basename, dirname as dr
from os import makedirs
from omegaconf import DictConfig, OmegaConf
import hydra

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from conv_ssl.callbacks import SymmetricSpeakersCallback
from conv_ssl.model import VPModel
from conv_ssl.utils import (
    everything_deterministic,
    write_json,
    read_json,
    tensor_dict_to_json,
)
from datasets_turntaking import DialogAudioDM

# ugly path

SAVEPATH = join(dr(dr(dr(__file__))), "assets/PaperB/eval")
MIN_THRESH = 0.01  # minimum threshold limit for S/L, S-pred, BC-pred

everything_deterministic()

"""
python conv_ssl/evaluation/evaluation.py \
        +checkpoint_path=/FULL/PATH/TO/CHECKPOINT/checkpoint.ckpt \
        data.num_workers=4 \
        data.batch_size=10
"""


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
        print("Num Workers: ", data_conf["num_workers"])
        print("Batch size: ", data_conf["batch_size"])
        print("Mono: ", data_conf["audio_mono"])
        print("datasets: ", data_conf["datasets"])
    dm = DialogAudioDM(**data_conf)
    dm.prepare_data()
    dm.setup("test")


def test(model, dloader, max_batches=None, project="VPModelTest", online=False):
    """
    Iterate over the dataloader to extract metrics.

    * Adds SymmetricSpeakersCallback
        - each sample is duplicated with channels reversed
    * online = True
        - upload to wandb
    """
    logger = None
    if online:
        savedir = "runs/" + project
        makedirs(savedir, exist_ok=True)
        logger = WandbLogger(
            save_dir=savedir,
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
            callbacks=[SymmetricSpeakersCallback()],
        )
    else:
        trainer = Trainer(
            gpus=-1,
            deterministic=True,
            logger=logger,
            callbacks=[SymmetricSpeakersCallback()],
        )

    result = trainer.test(model, dataloaders=dloader, verbose=False)
    return result


def get_curves(preds, target, pos_label=1, thresholds=None, EPS=1e-6):
    """
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)

    """

    if thresholds is None:
        thresholds = torch.linspace(0, 1, steps=101)

    if pos_label == 0:
        raise NotImplemented("Have not done this")

    ba, f1 = [], []
    auc0, auc1 = [], []
    prec0, rec0 = [], []
    prec1, rec1 = [], []
    pos_label_idx = torch.where(target == 1)
    neg_label_idx = torch.where(target == 0)

    for t in thresholds:
        pred_labels = (preds >= t).float()
        correct = pred_labels == target

        # POSITIVES
        tp = correct[pos_label_idx].sum()
        n_p = (target == 1).sum()
        fn = n_p - tp
        # NEGATIVES
        tn = correct[neg_label_idx].sum()
        n_n = (target == 0).sum()
        fp = n_n - tn
        ###################################3
        # Balanced Accuracy
        ###################################3
        # TPR, TNR
        tpr = tp / n_p
        tnr = tn / n_n
        # BA
        ba_tmp = (tpr + tnr) / 2
        ba.append(ba_tmp)
        ###################################3
        # F1
        ###################################3
        precision1 = tp / (tp + fp + EPS)
        recall1 = tp / (tp + fn + EPS)
        f1_1 = 2 * precision1 * recall1 / (precision1 + recall1 + EPS)
        prec1.append(precision1)
        rec1.append(recall1)
        auc1.append(precision1 * recall1)

        precision0 = tn / (tn + fn + EPS)
        recall0 = tn / (tn + fp + EPS)
        f1_0 = 2 * precision0 * recall0 / (precision0 + recall0 + EPS)
        prec0.append(precision0)
        rec0.append(recall0)
        auc0.append(precision0 * recall0)

        f1w = (f1_0 * n_n + f1_1 * n_p) / (n_n + n_p)
        f1.append(f1w)

    return {
        "bacc": torch.stack(ba),
        "f1": torch.stack(f1),
        "prec1": torch.stack(prec1),
        "rec1": torch.stack(rec1),
        "prec0": torch.stack(prec0),
        "rec0": torch.stack(rec0),
        "auc0": torch.stack(auc0),
        "auc1": torch.stack(auc1),
        "thresholds": thresholds,
    }


def find_threshold(model, dloader, min_thresh=0.01):
    """Find the best threshold using PR-curves"""

    def get_best_thresh(curves, metric, measure, min_thresh):
        ts = curves[metric]["thresholds"]
        over = min_thresh <= ts
        under = ts <= (1 - min_thresh)
        w = torch.where(torch.logical_and(over, under))
        values = curves[metric][measure][w]
        ts = ts[w]
        _, best_idx = values.max(0)
        return ts[best_idx]

    # Init metric:
    model.test_metric = model.init_metric(
        bc_pred_pr_curve=True,
        shift_pred_pr_curve=True,
        long_short_pr_curve=True,
    )

    # Find Thresholds
    _ = test(model, dloader, online=False)

    ############################################
    # Save predictions
    predictions = {}
    if hasattr(model.test_metric, "long_short_pr"):
        predictions["long_short"] = {
            "preds": torch.cat(model.test_metric.long_short_pr.preds),
            "target": torch.cat(model.test_metric.long_short_pr.target),
        }
    if hasattr(model.test_metric, "bc_pred_pr"):
        predictions["bc_preds"] = {
            "preds": torch.cat(model.test_metric.bc_pred_pr.preds),
            "target": torch.cat(model.test_metric.bc_pred_pr.target),
        }
    if hasattr(model.test_metric, "shift_pred_pr"):
        predictions["shift_preds"] = {
            "preds": torch.cat(model.test_metric.shift_pred_pr.preds),
            "target": torch.cat(model.test_metric.shift_pred_pr.target),
        }

    ############################################
    # Curves
    curves = {}
    for metric in ["bc_preds", "long_short", "shift_preds"]:
        curves[metric] = get_curves(
            preds=predictions[metric]["preds"], target=predictions[metric]["target"]
        )

    ############################################
    # find best thresh
    bc_pred_threshold = None
    shift_pred_threshold = None
    long_short_threshold = None
    if "bc_preds" in curves:
        bc_pred_threshold = get_best_thresh(curves, "bc_preds", "f1", min_thresh)
    if "shift_preds" in curves:
        shift_pred_threshold = get_best_thresh(curves, "shift_preds", "f1", min_thresh)
    if "long_short" in curves:
        long_short_threshold = get_best_thresh(curves, "long_short", "f1", min_thresh)

    thresholds = {
        "pred_shift": shift_pred_threshold,
        "pred_bc": bc_pred_threshold,
        "short_long": long_short_threshold,
    }
    return thresholds, predictions, curves


@hydra.main(config_path="../conf", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate model"""
    cfg_dict = OmegaConf.to_object(cfg)
    cfg_dict = dict(cfg_dict)

    # Load model
    model = VPModel.load_from_checkpoint(cfg.checkpoint_path, strict=False)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    savepath = join(SAVEPATH, basename(cfg.checkpoint_path).replace(".ckpt", ""))
    savepath += "_" + "_".join(cfg.data.datasets)
    Path(savepath).mkdir(exist_ok=True, parents=True)

    # Load data
    print("Num Workers: ", cfg.data.num_workers)
    print("Batch size: ", cfg.data.batch_size)
    print(cfg.data.datasets)
    dm = DialogAudioDM(
        datasets=cfg.data.datasets,
        type=cfg.data.type,
        audio_duration=cfg.data.audio_duration,
        audio_normalize=cfg.data.audio_normalize,
        audio_overlap=cfg.data.audio_overlap,
        sample_rate=cfg.data.sample_rate,
        vad_hz=model.frame_hz,
        vad_horizon=model.VAP.horizon,
        vad_history=cfg.data.vad_history,
        vad_history_times=cfg.data.vad_history_times,
        flip_channels=False,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    dm.prepare_data()
    dm.setup(None)

    # Threshold
    # Find the best thresholds (S-pred, BC-pred, S/L) on the validation set
    threshold_path = cfg.get("thresholds", None)
    if threshold_path is None:
        print("#" * 60)
        print("Finding Thresholds (val-set)...")
        print("#" * 60)
        thresholds, prediction, curves = find_threshold(
            model, dm.val_dataloader(), min_thresh=MIN_THRESH
        )

        th = {k: v.item() for k, v in thresholds.items()}
        write_json(th, join(savepath, "thresholds.json"))
        torch.save(prediction, join(savepath, "predictions.pt"))
        torch.save(curves, join(savepath, "curves.pt"))
        print("Saved Thresholds -> ", join(savepath, "thresholds.json"))
        print("Saved Curves -> ", join(savepath, "curves.pt"))
    else:
        print("Loading thresholds: ", threshold_path)
        thresholds = read_json(threshold_path)

    # Score
    print("#" * 60)
    print("Final Score (test-set)...")
    print("#" * 60)
    model.test_metric = model.init_metric(
        threshold_pred_shift=thresholds.get("pred_shift", 0.5),
        threshold_short_long=thresholds.get("short_long", 0.5),
        threshold_bc_pred=thresholds.get("pred_bc", 0.5),
    )
    result = test(model, dm.test_dataloader(), online=False)[0]
    metrics = model.test_metric.compute()

    metrics["loss"] = result["test_loss"]
    metrics["threshold_pred_shift"] = thresholds["pred_shift"]
    metrics["threshold_pred_bc"] = thresholds["pred_bc"]
    metrics["threshold_short_long"] = thresholds["short_long"]

    metric_json = tensor_dict_to_json(metrics)
    write_json(metric_json, join(savepath, "metric.json"))
    print("Saved metrics -> ", join(savepath, "metric.pt"))


if __name__ == "__main__":
    evaluate()
