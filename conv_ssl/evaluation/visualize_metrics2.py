from os.path import join, dirname
from os import makedirs
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

import torch
import pytorch_lightning as pl
from torchmetrics import PrecisionRecallCurve

from conv_ssl.plot_utils import plot_curve, plot_batch
from conv_ssl.utils import everything_deterministic, to_device
from conv_ssl.evaluation.evaluation import test
from conv_ssl.evaluation.k_fold_aggregate import (
    discrete,
    independent,
    independent_baseline,
    test_single_model,
)
from conv_ssl.evaluation.utils import load_dm, load_model

event_kwargs = dict(
    shift_onset_cond=1,
    shift_offset_cond=1,
    hold_onset_cond=1,
    hold_offset_cond=1,
    min_silence=0.15,
    non_shift_horizon=2.0,
    non_shift_majority_ratio=0.95,
    metric_pad=0.05,
    metric_dur=0.1,
    metric_onset_dur=0.2,
    metric_pre_label_dur=0.5,
    metric_min_context=1.0,
    min_context=3.0,
    bc_max_duration=1.0,
    bc_pre_silence=1.0,
    bc_post_silence=2.0,
)


def plot_all_curves(c, min_thresh, title, figsize=(9, 6), plot=False):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax = plot_curve(
        c["f1"],
        c["thresholds"],
        min_thresh=min_thresh,
        max_thresh=1 - min_thresh,
        label="F1 ",
        ax=ax,
        color="b",
    )
    ax = plot_curve(
        c["bacc"],
        c["thresholds"],
        min_thresh=min_thresh,
        max_thresh=1 - min_thresh,
        label="BAcc ",
        ax=ax,
        color="r",
        plot_guide_lines=True,
    )
    ax = plot_curve(
        c["auc1"],
        c["thresholds"],
        min_thresh=min_thresh,
        max_thresh=1 - min_thresh,
        label="AUC1 ",
        ax=ax,
        color="g",
    )
    ax = plot_curve(
        c["auc0"],
        c["thresholds"],
        min_thresh=min_thresh,
        max_thresh=1 - min_thresh,
        label="AUC0: ",
        ax=ax,
        color="darkgreen",
    )
    ax.set_ylabel("F1", fontsize=18)
    ax.set_xlabel("Threshold", fontsize=18)
    ax.set_ylim([0, 1.02])
    ax.set_title(title)
    ax.legend()
    if plot:
        plt.pause(0.1)
        # plt.show()
    return fig, ax


def test_single_model(
    run_path,
    event_kwargs,
    threshold_pred_shift=0.5,
    threshold_short_long=0.5,
    threshold_bc_pred=0.1,
    bc_pred_pr_curve=False,
    shift_pred_pr_curve=False,
    long_short_pr_curve=False,
    batch_size=16,
):
    """test model"""

    # Load data (same across folds)
    dm = load_dm(vad_hz=100, horizon=2, batch_size=batch_size, num_workers=4)

    # Load model and process test-set
    model = load_model(run_path=run_path, eval=True, strict=False)

    if torch.cuda.is_available():
        model = model.to("cuda")

    # Updatemetric_kwargs metrics
    # for metric, val in metric_kwargs.items():
    #     model.conf["vad_projection"][metric] = val

    model.test_metric = model.init_metric(
        model.conf,
        model.frame_hz,
        threshold_pred_shift=threshold_pred_shift,
        threshold_short_long=threshold_short_long,
        threshold_bc_pred=threshold_bc_pred,
        bc_pred_pr_curve=bc_pred_pr_curve,
        shift_pred_pr_curve=shift_pred_pr_curve,
        long_short_pr_curve=long_short_pr_curve,
        **event_kwargs,
    )
    model.test_metric = model.test_metric.to(model.device)

    result = test(model, dm, online=False)
    return result, model


def batch_view():
    # Model
    run_path = discrete["0"]
    # run_path = independent["4"]
    model = load_model(run_path=run_path, eval=True, strict=False)
    model = model.eval()
    model.test_metric = model.init_metric(
        model.conf, model.frame_hz, bc_pred_pr_curve=False, **event_kwargs
    )
    model.test_metric = model.test_metric.to(model.device)

    dm = load_dm(batch_size=16)
    diter = iter(dm.test_dataloader())
    batch = next(diter)
    batch = to_device(batch)
    # BATCH
    with torch.no_grad():
        events = model.test_metric.extract_events(batch["vad"])
        _, out, batch = model.shared_step(batch)
        probs = model.get_next_speaker_probs(out["logits_vp"], batch["vad"])
        for k, v in events.items():
            events[k] = v[:, :1000]
        # for k, v in events.items():
        #     print(k, v.shape)
    fig, ax = plot_batch(probs, events, batch["vad"], metric="predict_bc")
    fig, ax = plot_batch(probs, events, batch["vad"], metric="predict_shift")
    fig, ax = plot_batch(probs, events, batch["vad"], metric="long")


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
        try:
            correct = pred_labels == target
        except:
            print("pred_labels: ", tuple(pred_labels.shape))
            print("target: ", tuple(target.shape))
            input()

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


def extract_loss_curve():
    from conv_ssl.evaluation.evaluation import SymmetricSpeakersCallback

    run_path = discrete["3"]
    # Load data (same across folds)
    dm = load_dm(vad_hz=100, horizon=2, batch_size=16, num_workers=4)
    # Load model and process test-set
    model = load_model(run_path=run_path, eval=True, strict=False)
    # if torch.cuda.is_available():
    #     model = model.to("cuda")

    # model.test_metric = model.init_metric(
    #     model.conf,
    #     model.frame_hz,
    #     # threshold_pred_shift=threshold_pred_shift,
    #     # threshold_short_long=threshold_short_long,
    #     # threshold_bc_pred=threshold_bc_pred,
    #     # bc_pred_pr_curve=bc_pred_pr_curve,
    #     # shift_pred_pr_curve=shift_pred_pr_curve,
    #     # long_short_pr_curve=long_short_pr_curve,
    #     **event_kwargs,
    # )
    # model.test_metric = model.test_metric.to(model.device)

    trainer = pl.Trainer(
        gpus=-1,
        deterministic=True,
        logger=None,
        callbacks=[SymmetricSpeakersCallback()],
    )
    result = trainer.test(model, dataloaders=dm.val_dataloader(), verbose=False)

    loss_vector = model.loss_vector / model.loss_n

    # torch.save(loss_vector, "loss_vector.pt")
    # loss_vector = torch.load('loss_vector.pt')
    loss_vector = torch.load("loss_vector_full.pt")

    torch.save(loss_vector, "loss_vector_0.pt")

    # n = 0
    # losses = torch.zeros(1000, dtype=torch.float)
    # dloader = dm.val_dataloader()
    # for batch in tqdm(dloader, total=len(dloader)):
    #     loss, _, _ = model.shared_step(to_device(batch, model.device), reduction="none")
    #     n += loss["vp"].shape[0]
    #     losses += loss["vp"].sum(0).cpu()
    #     print(losses.device)
    # losses /= n

    # loss_vector = torch.load("loss_vector_full.pt")
    # ind = False
    loss_vector = torch.load("loss_vector_full_ind.pt")
    ind = True

    M = loss_vector.mean()
    if ind:
        m = loss_vector.mean(dim=0).mean(dim=-1).mean(dim=-1)
        # average over bins and speaker
        s = loss_vector.mean(dim=-1).mean(dim=-1)
        S = s.std(unbiased=True)
        s = s.std(dim=0, unbiased=True)
    else:
        S = loss_vector.std(unbiased=True)
        m = loss_vector.mean(dim=0)
        s = loss_vector.std(dim=0, unbiased=True)
    fig, ax = plt.subplots(1, 1)
    ax.plot(m, alpha=0.8, label="avg(t)", color="b")
    ax.plot(s, color="r", alpha=0.8, label="std(t)")
    ax.hlines(y=M, xmin=0, xmax=1000, label="Average", color="darkblue", linewidth=3)
    ax.hlines(y=S, xmin=0, xmax=1000, label="Std", color="darkred", linewidth=3)
    if ind:
        ax.set_ylabel("BCE Loss")
    else:
        ax.set_ylabel("Cross Entropy Loss")
    ax.set_xlabel("time step (10ms)")
    ax.legend(loc="upper right")
    plt.pause(0.1)



def eval_single_model(
    model_name,
    kfold,
    min_thresh=0.01,
    project_id="how_so/VPModel",
    verbose=False,
    plot=False,
):
    """
    # model_name = "independent"
    # kfold = 4
    # model_name = 'discrete'
    # kfold = 5
    """

    if model_name == "discrete":
        id = discrete[str(kfold)]
    elif model_name == "independent":
        id = independent[str(kfold)]
    elif model_name == "independent_baseline":
        id = independent_baseline[str(kfold)]
    else:
        raise NotImplementedError("")

    root = "assets/score"
    makedirs(root, exist_ok=True)
    predictions_path = f"{root}/{model_name}/kfold{kfold}_{model_name}_predictions.pt"
    curve_path = f"{root}/{model_name}/kfold{kfold}_{model_name}_curves.pt"
    run_path = join(project_id, id)

    if verbose:
        print("run_path: ", run_path)
        print("model_name: ", model_name)
        print("kfold: ", kfold)
        print("predictions_path: ", predictions_path)
        print("curve_path: ", curve_path)

    # Aggregate
    _, model = test_single_model(
        run_path,
        event_kwargs=event_kwargs,
        threshold_pred_shift=0.5,
        threshold_short_long=0.5,
        threshold_bc_pred=0.1,
        bc_pred_pr_curve=True,
        shift_pred_pr_curve=True,
        long_short_pr_curve=True,
        batch_size=16,
    )

    ############################################
    # Save predictions
    predictions = {
        "long_short": {
            "preds": torch.cat(model.test_metric.long_short_pr.preds),
            "target": torch.cat(model.test_metric.long_short_pr.target),
        },
        "bc_preds": {
            "preds": torch.cat(model.test_metric.bc_pred_pr.preds),
            "target": torch.cat(model.test_metric.bc_pred_pr.target),
        },
        "shift_preds": {
            "preds": torch.cat(model.test_metric.shift_pred_pr.preds),
            "target": torch.cat(model.test_metric.shift_pred_pr.target),
        },
        "kfold": kfold,
        "model": model.run_name,
        "name": model_name,
    }
    makedirs(dirname(predictions_path), exist_ok=True)
    torch.save(predictions, predictions_path)

    ############################################
    # Save Curves
    curves = {}
    titles = [
        f'BC prediction: {predictions["name"]}',
        f'Short/Long Onset 1/0: {predictions["name"]}',
        f'SHIFT prediction: {predictions["name"]}',
    ]
    metrics = ["bc_preds", "long_short", "shift_preds"]
    for title, metric in zip(titles, metrics):
        preds, target = predictions[metric]["preds"], predictions[metric]["target"]
        c = get_curves(preds, target)
        curves[metric] = c
        if plot:
            fig, ax = plot_all_curves(c, min_thresh, title, figsize=(9, 6), plot=plot)
    makedirs(dirname(curve_path), exist_ok=True)
    torch.save(curves, curve_path)

    ############################################
    # find best thresh

    def get_best_thresh(metric, measure):
        ts = curves[metric]["thresholds"]
        over = min_thresh <= ts
        under = ts <= (1 - min_thresh)
        w = torch.where(torch.logical_and(over, under))
        values = curves[metric][measure][w]
        ts = ts[w]
        best, best_idx = values.max(0)
        return ts[best_idx]

    long_short_threshold = get_best_thresh("long_short", "f1")
    shift_pred_threshold = get_best_thresh("shift_preds", "f1")
    bc_pred_threshold = get_best_thresh("bc_preds", "f1")

    #############################################
    # get scores
    # Aggregate
    res, model = test_single_model(
        run_path,
        event_kwargs=event_kwargs,
        threshold_pred_shift=shift_pred_threshold,
        threshold_short_long=long_short_threshold,
        threshold_bc_pred=bc_pred_threshold,
        bc_pred_pr_curve=False,
        shift_pred_pr_curve=False,
        long_short_pr_curve=False,
        batch_size=16,
    )
    res = res[0]

    return {
        "loss": res["test_loss"],
        "f1_hold_shift": res["test/f1_hold_shift"],
        "f1_predict_shift": res["test/f1_predict_shift"],
        "f1_short_long": res["test/f1_short_long"],
        "f1_bc_prediction": res["test/f1_bc_prediction"],
        "threshold_pred_shift": shift_pred_threshold,
        "threshold_short_long": long_short_threshold,
        "threshold_bc_pred": bc_pred_threshold,
        "model_name": model_name,
        "predictions_path": predictions_path,
        "curve_path": curve_path,
    }


def debug_curves():
    # LOAD SCORE

    all_score = torch.load("assets/score/all_score_new.pt")

    scores = {
        "loss": [],
        "f1_hold_shift": [],
        "f1_predict_shift": [],
        "f1_short_long": [],
        "f1_bc_prediction": [],
        "threshold_pred_shift": [],
        "threshold_short_long": [],
        "threshold_bc_pred": [],
    }
    for metric in scores.keys():
        for model, model_scores in all_score.items():
            t = torch.tensor(model_scores[metric])
            scores[metric].append(t.mean())
    f1_metrics = [
        "f1_hold_shift",
        "f1_predict_shift",
        "f1_bc_prediction",
        "f1_short_long",
    ]
    models = {}
    for model, model_scores in all_score.items():
        models[model] = {"mean": [], "std": [], "thresh": {"mean": [], "std": []}}
        for metric in f1_metrics:
            t = torch.tensor(model_scores[metric])
            models[model]["mean"].append(t.mean())
            models[model]["std"].append(t.std())
            thresh = None
            if "short" in metric:
                thresh = torch.tensor(model_scores["threshold_short_long"])
            elif "predict_shift" in metric:
                thresh = torch.tensor(model_scores["threshold_pred_shift"])
            elif "bc_prediction" in metric:
                thresh = torch.tensor(model_scores["threshold_bc_pred"])
            if thresh is not None:
                models[model]["thresh"]["mean"].append(thresh.mean())
                models[model]["thresh"]["std"].append(thresh.std())

    import plotly.graph_objects as go

    metrics = ["f1_hold_shift", "f1_predict_shift", "f1_bc_prediction", "f1_short_long"]
    fig = go.Figure(
        data=[
            go.Bar(name="Discrete", x=metrics, y=models["discrete"]["mean"]),
            go.Bar(name="Independent", x=metrics, y=models["independent"]["mean"]),
            go.Bar(
                name="Independent-baseline",
                x=metrics,
                y=models["independent_baseline"]["mean"],
            ),
        ]
    )
    fig.update_layout(barmode="group")
    fig.show()
    metrics = ["f1_predict_shift", "f1_bc_prediction", "f1_short_long"]
    fig = go.Figure(
        data=[
            go.Bar(name="Discrete", x=metrics, y=models["discrete"]["thresh"]["mean"]),
            go.Bar(
                name="Independent", x=metrics, y=models["independent"]["thresh"]["mean"]
            ),
            go.Bar(
                name="Independent-baseline",
                x=metrics,
                y=models["independent_baseline"]["thresh"]["mean"],
            ),
        ]
    )
    fig.update_layout(barmode="group")
    fig.show()

    # SINGLE
    # preds = torch.load("assets/score/discrete/kfold5_discrete_predictions.pt")
    # curves = torch.load("assets/score/discrete/kfold5_discrete_curves.pt")
    # model_name = 'Discrete'
    preds = torch.load("assets/score/independent/kfold4_independent_predictions.pt")
    curves = torch.load("assets/score/independent/kfold4_independent_curves.pt")
    model_name = "Independent"
    # fig, ax = plot_all_curves(curves['bc_preds'], min_thresh=0.01, title=f'BC pred: {model_name}', figsize=(9, 6), plot=True)
    # fig, ax = plot_all_curves(curves['shift_preds'], min_thresh=0.01, title=f'Shift pred: {model_name}', figsize=(9, 6), plot=True)
    fig, ax = plot_all_curves(
        curves["long_short"],
        min_thresh=0.01,
        title=f"Long/Short: {model_name}",
        figsize=(9, 6),
        plot=True,
    )


def extract_threshold_and_result():
    t = time.time()
    all_score = {}
    for model in ["discrete", "independent", "independent_baseline"]:
        scores = {}
        for kfold in range(11):
            score = eval_single_model(model_name="discrete", kfold=kfold, verbose=True)
            for k, v in score.items():
                if k not in scores:
                    scores[k] = [v]
                else:
                    scores[k].append(v)
        all_score[model] = scores
    t = time.time() - t
    torch.save(all_score, "assets/score/all_score.pt")
    print(f"Time: {round(t, 2)}s")


def ANOVA():
    import scipy.stats as stats
    import numpy as np
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # all_score = torch.load("assets/score/all_score_new.pt")
    all_score = torch.load("assets/score/all_score.pt")
    result = {}
    for model in ["discrete", "independent", "independent_baseline"]:
        result[model] = {
            "SH": round(
                torch.tensor(all_score[model]["f1_hold_shift"]).mean().item(), 3
            ),
            "SL": round(
                torch.tensor(all_score[model]["f1_short_long"]).mean().item(), 3
            ),
            "S-pred": round(
                torch.tensor(all_score[model]["f1_predict_shift"]).mean().item(), 3
            ),
            "BC-pred": round(
                torch.tensor(all_score[model]["f1_bc_prediction"]).mean().item(), 3
            ),
        }
    statistics = {
        "SH": {},
        "SL": {},
        "S-pred": {},
        "BC-pred": {},
    }
    for ii, (metric, new_metric) in enumerate(
        zip(
            ["f1_hold_shift", "f1_short_long", "f1_predict_shift", "f1_bc_prediction"],
            ["SH", "SL", "S-pred", "BC-pred"],
        )
    ):
        m_discrete = all_score["discrete"][metric]
        m_ind = all_score["independent"][metric]
        m_ind_base = all_score["independent_baseline"][metric]
        anova_result = stats.f_oneway(m_discrete, m_ind, m_ind_base)
        statistics[new_metric] = anova_result.pvalue
        # ad-Hoc test
        d_vs_i = stats.ttest_ind(m_discrete, m_ind)
        d_vs_ib = stats.ttest_ind(m_discrete, m_ind_base)
        statistics[f"{new_metric}_t_test_d_vs_i"] = d_vs_i.pvalue
        statistics[f"{new_metric}_t_test_d_vs_ib"] = d_vs_ib.pvalue
    for model, row in result.items():
        print(model, row)
    print("-" * 30)
    for metric, pval in statistics.items():
        print(metric, pval)

    # print(result)
    print(statistics)

    for model, scores in all_score.items():
        print(model)
        for metric, val in scores.items():
            if "f1" in metric:
                # print(metric, torch.tensor(val).std())
                print(metric, round(torch.tensor(val).mean().item(), 3))
            # if "threshold" in metric:
            #     t = torch.stack(val)
            #     t0 = round(t.min().item(), 3)
            #     t1 = round(t.max().item(), 3)
            #     s = round(t.std().item(), 3)
            # print(f"{metric} ({t0}, {t1}), s={s}")
        print("-" * 30)

    fig, ax = plt.subplots(4, 1)
    for ii, metric in enumerate(
        ["f1_hold_shift", "f1_predict_shift", "f1_bc_prediction", "f1_short_long"]
    ):
        m_discrete = all_score["discrete"][metric]
        m_ind = all_score["independent"][metric]
        m_ind_base = all_score["independent_baseline"][metric]
        anova_result = stats.f_oneway(m_discrete, m_ind, m_ind_base)
        print(f"{metric} p={anova_result.pvalue} ({round(anova_result.statistic, 3)})")
        # Post-Hoc test
        d_vs_i = stats.ttest_ind(m_discrete, m_ind)
        d_vs_ib = stats.ttest_ind(m_discrete, m_ind_base)
        print(f"Discrete vs Independent t-test: {d_vs_i.pvalue}")
        print(f"Discrete vs Independent Base t-test: {d_vs_ib.pvalue}")
        print("-" * 50)
        # Tukey
        endog = m_discrete + m_ind + m_ind_base
        groups = (
            ["discrete"] * len(m_discrete)
            + ["ind"] * len(m_ind)
            + ["ind_base"] * len(m_ind_base)
        )
        tukey = pairwise_tukeyhsd(
            endog=endog, groups=groups, alpha=0.05
        )  # Significance level
        tukey.plot_simultaneous(ax=ax[ii])  # Plot group confidence intervals
        ax[ii].set_ylabel(metric)
        xmin, xmax = ax[ii].get_xlim()
        xmid = xmin + (xmax - xmin) / 2
        ax[ii].set_xlim([xmid - 0.04, xmid + 0.04])
        ax[ii].set_title("")
    plt.tight_layout()
    plt.show()


def majority_class_eval():
    from vad_turn_taking.metrics import TurnTakingMetrics

    # Load data (same across folds)
    frame_hz = 100
    dm = load_dm(vad_hz=frame_hz, horizon=2, batch_size=16, num_workers=4)

    def get_majority_class_probs(events):
        # We want to guess all holds
        p = torch.zeros_like(events["shift"])
        wh = torch.where(events["hold"])
        p[wh] = torch.ones(len(wh[0]), device=p.device, dtype=torch.float)
        wh = torch.where(events["shift"])
        p[wh] = torch.zeros(len(wh[0]), device=p.device, dtype=torch.float)

        # predict shift -> always predict hold
        w = torch.where(events["predict_shift_pos"])
        p[w] = torch.zeros(len(w[0]), device=p.device, dtype=torch.float)
        w = torch.where(events["predict_shift_neg"])
        p[w] = torch.zeros(len(w[0]), device=p.device, dtype=torch.float)

        # BC / Short vs long
        bc_probs = torch.zeros_like(events["short"])

        # always predict short
        w = torch.where(events["short"])
        bc_probs[w] = torch.ones(len(w[0]), device=bc_probs.device, dtype=torch.float)
        w = torch.where(events["long"])
        bc_probs[w] = torch.zeros(len(w[0]), device=bc_probs.device, dtype=torch.float)

        # Predict bc: Never predict bc
        w = torch.where(events["predict_bc_pos"])
        # bc_probs[w] = 0.
        bc_probs[w] = torch.zeros(len(w[0]), device=bc_probs.device, dtype=torch.float)
        w = torch.where(events["predict_bc_neg"])
        # bc_probs[w] = 0.
        bc_probs[w] = torch.zeros(len(w[0]), device=bc_probs.device, dtype=torch.float)
        return {"p": p, "bc_prediction": bc_probs}

    max_batch = 50
    metric = TurnTakingMetrics(
        threshold_pred_shift=0.5,
        threshold_short_long=0.5,
        threshold_bc_pred=0.5,
        bc_pred_pr_curve=False,
        shift_pred_pr_curve=False,
        long_short_pr_curve=False,
        frame_hz=frame_hz,
        **event_kwargs,
    )
    metric = metric.to("cuda")
    ii = 0
    for batch in tqdm(dm.test_dataloader()):
        batch = to_device(batch, device="cuda")
        events = metric.extract_events(batch["vad"], max_frame=1000)
        turn_taking_probs = get_majority_class_probs(events)
        metric.update(
            p=turn_taking_probs["p"],
            bc_pred_probs=turn_taking_probs["bc_prediction"],
            events=events,
        )
        ii += 1
        if ii == max_batch:
            break
    r = metric.compute()


if __name__ == "__main__":

    pl.seed_everything(100)
    everything_deterministic()

    # extract_loss_curve()
    extract_threshold_and_result()