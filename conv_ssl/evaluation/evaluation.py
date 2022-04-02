from os.path import join
from os import makedirs, cpu_count

import torch
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import WandbLogger

from conv_ssl.model import VPModel
from conv_ssl.utils import everything_deterministic, write_json

from datasets_turntaking import DialogAudioDM
from vap_turn_taking import TurnTakingMetrics


METRIC_CONF = dict(
    pad=0.05,  # int, pad on silence (shift/hold) onset used for evaluating\
    dur=0.1,  # int, duration off silence (shift/hold) used for evaluating\
    pre_label_dur=0.5,  # int, frames prior to Shift-silence for prediction on-active shift
    onset_dur=0.2,
    min_context=3.0,
)

HS_CONF = dict(
    post_onset_shift=1,
    pre_offset_shift=1,
    post_onset_hold=1,
    pre_offset_hold=1,
    non_shift_horizon=2,
    metric_pad=METRIC_CONF["pad"],
    metric_dur=METRIC_CONF["dur"],
    metric_pre_label_dur=METRIC_CONF["pre_label_dur"],
    metric_onset_dur=METRIC_CONF["onset_dur"],
)

BC_CONF = dict(
    max_duration_frames=1.0,
    pre_silence_frames=1.0,
    post_silence_frames=2.0,
    min_duration_frames=METRIC_CONF["onset_dur"],
    metric_dur_frames=METRIC_CONF["onset_dur"],
    metric_pre_label_dur=METRIC_CONF["pre_label_dur"],
)

MIN_THRESH = 0.01


def tensor_dict_to_json(d):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            v = v.tolist()
        elif isinstance(v, dict):
            v = tensor_dict_to_json(v)
        new_d[k] = v
    return new_d


class SymmetricSpeakersCallback(Callback):
    """
    This callback "flips" the speakers such that we get a fair evaluation not dependent on the
    biased speaker-order / speaker-activity

    The audio is mono which requires no change.

    The only change we apply is to flip the channels in the VAD-tensor and get the corresponding VAD-history
    which is defined as the ratio of speaker 0 (i.e. vad_history_flipped = 1 - vad_history)
    """

    def get_symmetric_batch(self, batch):
        """Appends a flipped version of the batch-samples"""
        for k, v in batch.items():
            if k == "vad":
                flipped = torch.stack((v[..., 1], v[..., 0]), dim=-1)
            elif k == "vad_history":
                flipped = 1.0 - v
            else:
                flipped = v
            if isinstance(v, torch.Tensor):
                batch[k] = torch.cat((v, flipped))
            else:
                batch[k] = v + flipped
        return batch

    def on_test_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.get_symmetric_batch(batch)

    def on_val_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.get_symmetric_batch(batch)


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
    model.test_metric = TurnTakingMetrics(
        hs_kwargs=HS_CONF,
        bc_kwargs=BC_CONF,
        metric_kwargs=METRIC_CONF,
        bc_pred_pr_curve=True,
        shift_pred_pr_curve=True,
        long_short_pr_curve=True,
        frame_hz=model.frame_hz,
    )
    model.test_metric.to(model.device)

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


def evaluate(checkpoint_path, savepath, batch_size, num_workers):
    """Evaluate model"""

    # Load model
    model = VPModel.load_from_checkpoint(checkpoint_path, strict=False)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    # Load data
    data_conf = DialogAudioDM.load_config()
    DialogAudioDM.print_dm(data_conf)
    print("Num Workers: ", num_workers)
    print("Batch size: ", batch_size)
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hz=model.frame_hz,
        vad_horizon=model.VAP.horizon,
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        flip_channels=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    dm.prepare_data()
    dm.setup(None)
    makedirs(savepath, exist_ok=True)

    # Threshold
    # Find the best thresholds (S-pred, BC-pred, S/L) on the validation set
    print("#" * 60)
    print("Finding Thresholds (val-set)...")
    print("#" * 60)
    thresholds, prediction, curves = find_threshold(
        model, dm.val_dataloader(), min_thresh=MIN_THRESH
    )
    torch.save(prediction, join(savepath, "predictions.pt"))
    torch.save(curves, join(savepath, "curves.pt"))

    # Score
    print("#" * 60)
    print("Final Score (test-set)...")
    print("#" * 60)
    model.test_metric = TurnTakingMetrics(
        hs_kwargs=HS_CONF,
        bc_kwargs=BC_CONF,
        metric_kwargs=METRIC_CONF,
        threshold_pred_shift=thresholds.get("pred_shift", 0.5),
        threshold_short_long=thresholds.get("short_long", 0.5),
        threshold_bc_pred=thresholds.get("pred_bc", 0.5),
        frame_hz=model.frame_hz,
    )
    model.test_metric.to(model.device)
    result = test(model, dm.test_dataloader(), online=False)[0]
    metrics = model.test_metric.compute()

    metrics["loss"] = result["test_loss"]
    metrics["threshold_pred_shift"] = thresholds["pred_shift"]
    metrics["threshold_pred_bc"] = thresholds["pred_bc"]
    metrics["threshold_short_long"] = thresholds["short_long"]

    torch.save(metrics, join(savepath, "metric.pt"))
    metric_json = tensor_dict_to_json(metrics)
    write_json(metric_json, join(savepath, "metric.json"))
    return metrics, prediction, curves


if __name__ == "__main__":
    from conv_ssl.evaluation.utils import get_checkpoint, load_paper_versions

    everything_deterministic()
    model_id = "120k8fdv"
    checkpoint_path = get_checkpoint(run_path=model_id)
    checkpoint_path = load_paper_versions(checkpoint_path)

    metrics, prediction, curves = evaluate(
        model_id, savepath="metric_test", batch_size=16, num_workers=cpu_count()
    )
