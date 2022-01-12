import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import einops
from einops.layers.torch import Rearrange
from torchmetrics import F1

from datasets_turntaking.features.vad import VAD, VadProjection


class Utils:
    @staticmethod
    def get_topk_from_logits(logits, k):
        probs_vp = logits.softmax(dim=-1)
        return probs_vp.topk(k)

    @staticmethod
    def get_topk_acc(topk_idx, label):
        if topk_idx.ndim > label.ndim:
            label = label.unsqueeze(-1)
        K = topk_idx.shape[-1]
        correct = (topk_idx == label).float()
        acc = []
        for i in range(1, K + 1):
            s = (correct[..., :i].sum(dim=-1) > 0).float().mean()
            acc.append(s)
        acc = torch.stack(acc)
        return acc, label.nelement()

    @staticmethod
    def topk_acc_specific_frames(topk_ns, label_ns, where_onehot):
        """
        separate hold/shift topk accuracy using the `next_speaker` labels and predictions.

        If the model predicts the correct next-speaker for the `HOLD` frames (`where_onehot`) then
        the hold guess is correct. Symmetrically this is true for shifts as well.
        """

        # Check if relevant segments exists
        n = where_onehot.sum()
        if n == 0:
            return None, None

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
        topk_acc = []
        for i in range(1, K + 1):
            s = (correct_ns[..., :i].sum(dim=-1) > 0).float().mean()
            topk_acc.append(s)
        return torch.stack(topk_acc), n

    @staticmethod
    def get_f1_prediction_labels(topk_ns, label_ns, hold_one_hot, shift_one_hot):
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
        turn_label = torch.cat(turn_label).long()
        return pred, turn_label

    @staticmethod
    def filter_context(data, min_context_frames, device="cpu"):
        for k, v in data.items():
            data[k] = v[:, min_context_frames:].to(device)
        return data


class ProjectionMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.acc = {
            "hold": [],
            "shift": [],
            "class": [],
            "class_silence": [],
        }
        self.N = {
            "hold": [],
            "shift": [],
            "class": [],
            "class_silence": [],
        }
        # turn-shift/hold
        self.f1 = F1(num_classes=2, multiclass=True)
        self.f1_weighted = F1(num_classes=2, average="weighted", multiclass=True)

    def weighted_average(self, topk, n):
        # stack on batch dim
        a = torch.stack(topk)
        n = torch.tensor(n)
        if a.ndim == 1:
            scaled = n * a
            weighted_average = scaled.sum() / n.sum()
        else:
            # unsqueeze topk dim
            n = n.unsqueeze(-1)
            # scale values based on frequency
            scaled = n * a
            # weighted topk acc
            weighted_average = scaled.sum(dim=0) / n.sum()
        return weighted_average

    def update(self, m):
        """Update metrics (every batch)"""
        # Update F1: 0 element for greedy/top guess
        if "f1" in m:
            self.f1(m["f1"]["prediction"][..., 0], m["f1"]["label"])
            self.f1_weighted(m["f1"]["prediction"][..., 0], m["f1"]["label"])

        for metric in m.keys():
            if metric in ["f1", "loss"]:
                continue
            value = m[metric]["topk"]
            if value is not None:
                self.acc[metric].append(m[metric]["topk"])
                self.N[metric].append(m[metric]["n"])

    def compute(self):
        """Compute the aggregate metrics (on epoch end)"""
        ret = {}

        # F1 are handled by torchmetrics
        ret["f1"] = self.f1.compute()
        ret["f1_weighted"] = self.f1_weighted.compute()

        # iterate over metrics
        for metric in self.acc.keys():
            ret[metric] = self.weighted_average(self.acc[metric], self.N[metric])

        ret["hold_n"] = int(sum(self.N["hold"]))
        ret["shift_n"] = int(sum(self.N["shift"]))
        total = ret["hold_n"] + ret["shift_n"]
        ret["shift_ratio"] = ret["shift_n"] / total
        ret["hold_ratio"] = ret["hold_n"] / total
        return ret

    def __repr__(self):
        result = self.compute()
        s = "ProjectionMetrics\n"
        for metric, value in result.items():
            s += f"\t{metric}: {value}\n"
        return s


class ProjectionMetricCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_metric = ProjectionMetrics()
        self.val_metric = ProjectionMetrics()
        self.test_metric = ProjectionMetrics()

    def _log(self, result, pl_module, split="train"):
        # log result
        for metric, value in result.items():
            if "_n" in metric or "_ratio" in metric:
                continue
            if isinstance(value, torch.Tensor) and value.nelement() > 1:
                for k, kval in enumerate(value, start=1):
                    pl_module.log(f"{split}/{metric}/topk_{k}", kval)
            else:
                pl_module.log(f"{split}/{metric}", value)

    def on_validation_epoch_start(self, *args, **kwargs):
        self.val_metric.reset()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.val_metric.update(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        result = self.val_metric.compute()
        self._log(result, pl_module, split="val")

    def on_test_epoch_start(self, *args, **kwargs):
        self.test_metric.reset()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.test_metric.update(outputs)

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

    def prepare_discrete(self, logits, vad, vad_label, k=5):
        """
        Prepare metrics for discrete vadprojection classes.
        """
        # Store data before removing 'min_context_frames'. We want the full data
        # in order to get next_speaker/hold/shift
        data = {"vad": vad, "vad_label": vad_label}
        data["hold_one_hot"], data["shift_one_hot"] = VAD.get_hold_shift_onehot(vad)
        data["topk_probs"], data["topk_idx"] = Utils.get_topk_from_logits(logits, k)
        data["topk_next_speaker"] = self.vad_projection_codebook.get_next_speaker(
            data["topk_idx"]
        )
        data["label_next_speaker"] = self.vad_projection_codebook.get_next_speaker(
            data["vad_label"]
        )
        return data

    def regression_to_discrete(self, logits):
        """
        logits -> sigmoid -> probs
        probs -> round -> onehot
        onehot -> discrete classes
        """
        return self.vad_projection_codebook.onehot_to_idx(logits.sigmoid().round())

    def prepare_regression(self, logits, vad, vad_label):
        """
        Prepare metrics for discrete vadprojection classes.
        regression values -> probs -> onehot -> discrete classes
        """
        # TODO:
        data = {"vad": vad, "vad_label": vad_label}
        data["hold_one_hot"], data["shift_one_hot"] = VAD.get_hold_shift_onehot(vad)
        data["topk_idx"] = self.regression_to_discrete(logits).unsqueeze(-1)
        data["topk_next_speaker"] = self.vad_projection_codebook.get_next_speaker(
            data["topk_idx"]
        )
        data["label_next_speaker"] = self.vad_projection_codebook.get_next_speaker(
            data["vad_label"]
        )
        return data

    def prepare_metrics(self, logits, vad, vad_label, min_context_frames=0, k=5):
        if self.regression:
            data = self.prepare_regression(logits, vad, vad_label)
        else:
            data = self.prepare_discrete(logits, vad, vad_label, k)

        # Remove frames that does not have enough context (and move to cpu)
        data = Utils.filter_context(data, min_context_frames, device="cpu")

        ######################################################################
        # find "nucleus" subset
        # p = 0.8
        # tp = topk_probs.cumsum(dim=-1)
        # tp = tp[tp <= 0.6]
        # tpk = topk_idx[: len(tp)]
        # top_p_speaker_probs = self.get_hold_shift_probs(vad=vad, probs=tp)

        ######################################################################
        # TopK for correct exact label classification
        class_topk, n = Utils.get_topk_acc(data["topk_idx"], data["vad_label"])
        # TopK only on silences
        silence_onehot = (data["vad"].sum(dim=-1) == 0).float()
        class_sil_topk, class_sil_n = Utils.topk_acc_specific_frames(
            data["topk_idx"], data["vad_label"], silence_onehot
        )

        # TopK hold/shift acc
        hold_topk, hold_n = Utils.topk_acc_specific_frames(
            data["topk_next_speaker"],
            data["label_next_speaker"],
            where_onehot=data["hold_one_hot"],
        )
        shift_topk, shift_n = Utils.topk_acc_specific_frames(
            data["topk_next_speaker"],
            data["label_next_speaker"],
            where_onehot=data["shift_one_hot"],
        )

        # Prediction/Label classes for F1 hold/shift metric
        f1_pred, f1_label = Utils.get_f1_prediction_labels(
            topk_ns=data["topk_next_speaker"],
            label_ns=data["label_next_speaker"],
            hold_one_hot=data["hold_one_hot"],
            shift_one_hot=data["shift_one_hot"],
        )
        return {
            "class": {"topk": class_topk, "n": n},
            "class_silence": {"topk": class_sil_topk, "n": class_sil_n},
            "hold": {"topk": hold_topk, "n": hold_n},
            "shift": {"topk": shift_topk, "n": shift_n},
            "f1": {"prediction": f1_pred, "label": f1_label},
        }

    def forward(self, x):
        return self.projection_head(x)


if __name__ == "__main__":
    D = 128
    head = ActivityProjectionHead(input_size=D, regression=True)
    x = torch.randn((4, 499, D))
    logits = head(x)
    print("logits: ", tuple(logits.shape))
