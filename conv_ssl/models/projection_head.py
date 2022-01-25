import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pytorch_lightning as pl
from torchmetrics import Metric, F1Score

from datasets_turntaking.features.vad import VAD, VadProjection


class ProjectionMetrics(Metric):
    """based on StatScores"""

    def __init__(
        self,
        k=5,
        min_context_frames=50,
        regression=False,
        bin_sizes=[20, 40, 60, 80],
        threshold_ratio=0.5,
        dist_sync_on_step: bool = False,
        **kwargs,
    ):
        # super().__init__(dist_sync_on_step=dist_sync_on_step, dist_sync_fn=self.dist_sync_fn, **kwargs)
        super().__init__(dist_sync_on_step=dist_sync_on_step, **kwargs)
        self.k = 1 if regression else k
        self.min_context_frames = min_context_frames

        self.regression = regression
        self.bin_sizes = bin_sizes
        self.vad_projection_codebook = VadProjection(
            n_bins=2 * len(bin_sizes),
            bin_sizes=bin_sizes,
            threshold_ratio=threshold_ratio,
        )

        self.f1 = F1Score(num_classes=2, multiclass=True)
        self.f1_weighted = F1Score(num_classes=2, average="weighted", multiclass=True)

        for turn_metric in ["hold", "shift"]:
            self.add_state(
                f"{turn_metric}_total", default=torch.tensor(0), dist_reduce_fx="sum"
            )
            self.add_state(
                f"{turn_metric}_acc",
                default=torch.zeros(k),
                dist_reduce_fx="sum",
            )

    def dist_sync_fn(self, outputs, **kwargs):
        print("outputs: ", outputs)
        for k, v in kwargs.items():
            print(f"kwarg: {k}: {v}")

    def regression_to_discrete(self, logits):
        """
        logits -> sigmoid -> probs
        probs -> round -> onehot
        onehot -> discrete classes
        """
        return self.vad_projection_codebook.onehot_to_idx(logits.sigmoid().round())

    def filter_context(self, data, device="cpu"):
        for k, v in data.items():
            data[k] = v[:, self.min_context_frames :]  # .to(device)
        return data

    def get_topk_from_logits(self, logits, k):
        probs_vp = logits.softmax(dim=-1)
        return probs_vp.topk(k)

    def topk_acc_specific_frames(self, topk_ns, label_ns, where_onehot):
        """
        separate hold/shift topk accuracy using the `next_speaker` labels and predictions.

        If the model predicts the correct next-speaker for the `HOLD` frames (`where_onehot`) then
        the hold guess is correct. Symmetrically this is true for shifts as well.
        """

        # Check if relevant segments exists
        n = where_onehot.sum()
        if n == 0:
            return torch.zeros((self.k,)), 0

        # k provided
        K = topk_ns.shape[-1]

        # where are frames for shift/hold
        ids = torch.where(where_onehot)
        y = label_ns[ids]  # next_speaker labels
        y_top = topk_ns[ids]  # predicted next speaker topk
        correct_ns = y.unsqueeze(-1) == y_top  # compare

        # Loop over the k to find if the model prediction is correct
        # in prediction 0 -> k
        # If the correct speaker is in the topk then we get a correct prediction
        # for that given k
        topk = []
        for i in range(1, K + 1):
            s = (correct_ns[..., :i].sum(dim=-1) > 0).float().sum()
            topk.append(s)
        return torch.stack(topk), n

    def get_f1_prediction_labels(self, topk_ns, label_ns, hold_one_hot, shift_one_hot):
        """
        From 'next speaker' prediction/labels we extract the hold/shift predictions based
        on `hold_one_hot`/`shift_one_hot`.

                speaker prediction    speaker label        f1_class_prediction  f1_label
        shifts: [0, 0, 0, 1, 1, 1]    [0, 0, 1, 1, 0, 0]   [1, 1, 0, 1, 0, 0]  [1, 1, 1, 1, 1, 1]
        holds : [0, 0, 0, 1, 1, 1]    [0, 0, 1, 1, 0, 0]   [0, 0, 1, 0, 1, 1]  [0, 0, 0, 0, 0, 0]

        preds: [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1]
        label: [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

        shift label = 1 and hold label = 0
        guessing correct next speaker inside of a shift -> 1 prediction
        guessing correct next speaker inside of a hold  -> 0 prediction
        """
        turn_label = []
        pred = []
        if hold_one_hot.sum() > 0:
            ids = torch.where(hold_one_hot)
            hold_lab = label_ns[ids]
            hold_pred = topk_ns[ids]
            # all correct preds (for hold) is set to 0 and incorrect to 1.
            hold_pred = (hold_pred != hold_lab.unsqueeze(-1)).float()
            pred.append(hold_pred)
            # Hold -> class = 0
            turn_label.append(torch.zeros(hold_one_hot.sum(), dtype=torch.long))

        if shift_one_hot.sum() > 0:
            ids = torch.where(shift_one_hot)
            shift_lab = label_ns[ids]
            shift_pred = topk_ns[ids]
            # all correct preds (for shift) is set to 1 and incorrect to 0.
            shift_pred = (shift_pred == shift_lab.unsqueeze(-1)).float()
            pred.append(shift_pred)
            # Shift -> class = 1
            turn_label.append(torch.ones(shift_one_hot.sum(), dtype=torch.long))

        if len(pred) == 0:
            return None, None

        pred = torch.cat(pred)
        turn_label = torch.cat(turn_label).long().to(pred.device)
        return pred, turn_label

    def update(self, logits, vad, vad_label):
        data = {"vad": vad, "vad_label": vad_label}
        data["hold_one_hot"], data["shift_one_hot"] = VAD.get_hold_shift_onehot(vad)

        # Regression metrics only has k=1
        if self.regression:
            data["topk_idx"] = self.regression_to_discrete(logits).unsqueeze(-1)
        else:
            data["topk_probs"], data["topk_idx"] = self.get_topk_from_logits(
                logits, k=self.k
            )

        data["topk_next_speaker"] = self.vad_projection_codebook.get_next_speaker(
            data["topk_idx"]
        )
        data["label_next_speaker"] = self.vad_projection_codebook.get_next_speaker(
            data["vad_label"]
        )

        # Remove frames that does not have enough context (and move to cpu)
        data = self.filter_context(data, device="cpu")

        # TopK hold/shift acc
        hold_topk, hold_n = self.topk_acc_specific_frames(
            data["topk_next_speaker"],
            data["label_next_speaker"],
            where_onehot=data["hold_one_hot"],
        )
        shift_topk, shift_n = self.topk_acc_specific_frames(
            data["topk_next_speaker"],
            data["label_next_speaker"],
            where_onehot=data["shift_one_hot"],
        )

        if self.hold_total.device != hold_n.device:
            self.to(hold_n.device)

        self.hold_total += hold_n
        self.shift_total += shift_n
        for i in range(self.k):
            self.hold_acc[i] += hold_topk[i]
            self.shift_acc[i] += shift_topk[i]

        # Prediction/Label classes for F1 hold/shift metric
        f1_pred, f1_label = self.get_f1_prediction_labels(
            topk_ns=data["topk_next_speaker"],
            label_ns=data["label_next_speaker"],
            hold_one_hot=data["hold_one_hot"],
            shift_one_hot=data["shift_one_hot"],
        )

        if f1_pred is not None and f1_label is not None:
            self.f1(f1_pred[..., 0], f1_label)
            self.f1_weighted(f1_pred[..., 0], f1_label)

    def compute(self):
        for i in range(self.k):
            self.hold_acc[i] /= self.hold_total
            self.shift_acc[i] /= self.shift_total

        return {
            "hold": {"acc": self.hold_acc, "n": self.hold_total},
            "shift": {"acc": self.shift_acc, "n": self.shift_total},
            "f1": self.f1.compute(),
            "f1_weighted": self.f1_weighted.compute(),
        }


class ProjectionMetricCallback(pl.Callback):
    def __init__(
        self,
        regression=False,
        bin_sizes=[20, 40, 60, 80],
        threshold_ratio=0.5,
        dist_sync_on_step=False,
    ):
        super().__init__()
        self.val_metric = ProjectionMetrics(
            k=5,
            min_context_frames=50,
            regression=regression,
            bin_sizes=bin_sizes,
            threshold_ratio=threshold_ratio,
            dist_sync_on_step=dist_sync_on_step,
        )
        self.test_metric = ProjectionMetrics(
            k=5,
            min_context_frames=50,
            regression=regression,
            bin_sizes=bin_sizes,
            threshold_ratio=threshold_ratio,
            dist_sync_on_step=dist_sync_on_step,
        )

    def _log(self, result, pl_module, split="train"):
        # log result
        for metric, value in result.items():
            if metric in ["hold", "shift"]:
                for k, k_score in enumerate(value["acc"]):
                    pl_module.log(
                        f"{split}/{metric}/topk_{k}", k_score, rank_zero_only=True
                    )
            else:
                pl_module.log(f"{split}/{metric}", value, rank_zero_only=True)

    def on_validation_epoch_start(self, *args, **kwargs):
        self.val_metric.reset()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        outputs = outputs["outputs"]
        self.val_metric.update(
            logits=outputs["logits_vp"],
            vad=outputs["batch"]["vad"],
            vad_label=outputs["batch"]["vad_label"],
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        result = self.val_metric.compute()
        self._log(result, pl_module=pl_module, split="val")

    def on_test_epoch_start(self, *args, **kwargs):
        self.test_metric.reset()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        outputs = outputs["outputs"]
        self.test_metric.update(
            logits=outputs["logits_vp"],
            vad=outputs["batch"]["vad"],
            vad_label=outputs["batch"]["vad_label"],
        )

    def on_test_epoch_end(self, trainer, pl_module):
        result = self.test_metric.compute()
        self._log(result, pl_module, split="test")


class ActivityProjectionHead(nn.Module):
    def __init__(
        self,
        input_size,
        bin_sizes=[20, 40, 60, 80],
        threshold_ratio=0.5,
        regression=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.regression = regression
        self.bin_sizes = bin_sizes
        self.total_bins = 2 * len(self.bin_sizes)
        self.threshold_ratio = threshold_ratio

        self.vad_projection_codebook = VadProjection(
            n_bins=self.total_bins,
            bin_sizes=self.bin_sizes,
            threshold_ratio=threshold_ratio,
        )

        if regression:
            self.projection_head = nn.Sequential(
                nn.Linear(input_size, self.total_bins),
                Rearrange("... (c f) -> ... c f", c=2, f=self.total_bins // 2),
            )
        else:
            self.projection_head = nn.Linear(
                input_size, self.vad_projection_codebook.n_classes
            )

    def loss_function(self, logits, labels, reduction="mean"):
        if self.regression:
            reg_labels = self.vad_projection_codebook(labels)
            loss = F.binary_cross_entropy_with_logits(
                logits, reg_labels, reduction=reduction
            )
        else:
            # CrossEntropyLoss over discrete labels
            loss = F.cross_entropy(
                einops.rearrange(logits, "b n d -> (b n) d"),
                einops.rearrange(labels, "b n -> (b n)"),
                reduction=reduction,
            )
            if reduction == "none":
                n = logits.shape[1]
                loss = einops.rearrange(loss, "(b n) -> b n", n=n)
        return loss

    def forward(self, x):
        return self.projection_head(x)


if __name__ == "__main__":
    D = 128
    head = ActivityProjectionHead(input_size=D, regression=True)
    x = torch.randn((4, 499, D))
    logits = head(x)
    print("logits: ", tuple(logits.shape))
