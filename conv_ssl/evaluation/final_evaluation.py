from os.path import join, dirname
from os import makedirs, cpu_count
from tqdm import tqdm

import torch
import scipy.stats as stats
import time

from conv_ssl.evaluation.evaluation import test
from conv_ssl.evaluation.utils import load_dm, load_model
from conv_ssl.utils import everything_deterministic, to_device
from conv_ssl.evaluation.k_fold_aggregate import (
    discrete,
    independent,
    independent_baseline,
    comparative,
)
from vad_turn_taking.metrics import TurnTakingMetrics

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
    num_workers=4,
    split="test",
):
    """test model"""

    # Load data (same across folds)
    dm = load_dm(vad_hz=100, horizon=2, batch_size=batch_size, num_workers=num_workers)

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

    result = test(model, dm, online=False, split=split)
    return result, model


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


def find_threshold(
    run_path,
    model_name,
    kfold,
    split,
    min_thresh,
    predictions_path,
    curve_path,
    bc_pred_pr_curve=True,
    shift_pred_pr_curve=True,
    long_short_pr_curve=True,
):
    def get_best_thresh(curves, metric, measure, min_thresh):
        ts = curves[metric]["thresholds"]
        over = min_thresh <= ts
        under = ts <= (1 - min_thresh)
        w = torch.where(torch.logical_and(over, under))
        values = curves[metric][measure][w]
        ts = ts[w]
        best, best_idx = values.max(0)
        return ts[best_idx]

    # Find Thresholds
    _, model = test_single_model(
        run_path,
        event_kwargs=event_kwargs,
        threshold_pred_shift=0.5,
        threshold_short_long=0.5,
        threshold_bc_pred=0.1,
        bc_pred_pr_curve=bc_pred_pr_curve,
        shift_pred_pr_curve=shift_pred_pr_curve,
        long_short_pr_curve=long_short_pr_curve,
        batch_size=16,
        num_workers=cpu_count(),
        split=split,
    )

    ############################################
    # Save predictions
    predictions = {
        "kfold": kfold,
        "model": model.run_name,
        "name": model_name,
    }

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
        try:
            preds, target = predictions[metric]["preds"], predictions[metric]["target"]
            c = get_curves(preds, target)
            curves[metric] = c
        except:
            pass

    makedirs(dirname(curve_path), exist_ok=True)
    torch.save(curves, curve_path)

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

    ret = {
        "shift": shift_pred_threshold,
        "bc": bc_pred_threshold,
        "long_short": long_short_threshold,
    }

    return ret


def get_threshold_and_eval(
    model_name,
    kfold,
    min_thresh=0.01,
    project_id="how_so/VPModel",
    verbose=False,
    plot=False,
    threshold_split="val",
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
    elif model_name == "comparative":
        id = comparative[str(kfold)]
    else:
        raise NotImplementedError("")

    root = "assets/score"
    makedirs(root, exist_ok=True)
    predictions_path = f"{root}/{model_name}/kfold{kfold}_{model_name}_predictions.pt"
    curve_path = f"{root}/{model_name}/kfold{kfold}_{model_name}_curves.pt"
    result_path = f"{root}/{model_name}/kfold{kfold}_{model_name}_result.pt"
    run_path = join(project_id, id)

    if verbose:
        print("run_path: ", run_path)
        print("model_name: ", model_name)
        print("kfold: ", kfold)
        print("predictions_path: ", predictions_path)
        print("curve_path: ", curve_path)

    #############################################
    # FIND THRESHOLD ON SOME SPLIT
    #############################################
    thresh = find_threshold(
        run_path=run_path,
        model_name=model_name,
        kfold=kfold,
        split=threshold_split,
        min_thresh=min_thresh,
        predictions_path=predictions_path,
        curve_path=curve_path,
    )

    #############################################
    # Final Test-set scores
    #############################################
    res, model = test_single_model(
        run_path,
        event_kwargs=event_kwargs,
        threshold_pred_shift=thresh["shift"],
        threshold_short_long=thresh["long_short"],
        threshold_bc_pred=thresh["bc"],
        bc_pred_pr_curve=False,
        shift_pred_pr_curve=False,
        long_short_pr_curve=False,
        batch_size=16,
        num_workers=cpu_count(),
        split="test",
    )
    res = res[0]
    torch.save(res, result_path)
    return {
        "loss": res["test_loss"],
        "f1_hold_shift": res["test/f1_hold_shift"],
        "f1_predict_shift": res["test/f1_predict_shift"],
        "f1_short_long": res["test/f1_short_long"],
        "f1_bc_prediction": res["test/f1_bc_prediction"],
        "threshold_pred_shift": thresh["shift"],
        "threshold_short_long": thresh["long_short"],
        "threshold_bc_pred": thresh["bc"],
        "model_name": model_name,
        "predictions_path": predictions_path,
        "curve_path": curve_path,
    }


# def comparative_threshold():


def Extract_Final_Scores():
    t = time.time()
    all_score = {}
    for model in ["independent", "independent_baseline", "discrete"]:
        scores = {}
        for kfold in range(11):
            score = get_threshold_and_eval(
                model_name=model, kfold=kfold, verbose=True, threshold_split="val"
            )
            for k, v in score.items():
                if k not in scores:
                    scores[k] = [v]
                else:
                    scores[k].append(v)
        all_score[model] = scores
    t = time.time() - t
    savepath = "assets/score/all_score_new_comp.pt"
    torch.save(all_score, savepath)
    print(f"Time: {round(t, 2)}s")
    print("Saved scores -> ", savepath)

    ##########################################################
    # Comparative
    comp_root = "assets/score/comp"
    comp_scores = {}
    for kfold, id in comparative.items():
        run_path = join("how_so/VPModel", id)
        thresh = find_threshold(
            run_path=run_path,
            model_name="comparative",
            kfold=kfold,
            split="val",
            min_thresh=0.01,
            predictions_path=f"{comp_root}/pred_{kfold}.pt",
            curve_path=f"{comp_root}/curve_{kfold}.pt",
            bc_pred_pr_curve=False,
            shift_pred_pr_curve=True,
            long_short_pr_curve=True,
        )
        res, model = test_single_model(
            run_path,
            event_kwargs=event_kwargs,
            threshold_pred_shift=thresh["shift"],
            threshold_short_long=thresh["long_short"],
            threshold_bc_pred=0,
            bc_pred_pr_curve=False,
            shift_pred_pr_curve=True,
            long_short_pr_curve=True,
            batch_size=16,
            num_workers=cpu_count(),
            split="test",
        )
        res = res[0]
        torch.save(res, f"{comp_root}/score_{kfold}.pt")
        for k, v in res.items():
            print(f"{k}: {v}")
            if k not in comp_scores:
                comp_scores[k] = [v]
            else:
                comp_scores[k].append(v)
        print(f"Finished kfold: {kfold}")
        print("#" * 60)
    torch.save(comp_scores, f"{comp_root}/all_score.pt")
    # all_score = {'comparative':torch.load("assets/score/comp/all_score.pt")}

    all_score = torch.load("assets/score/all_score_new.pt")
    all_score["comparative"] = {
        "loss": comp_scores["test_loss"],
        "f1_hold_shift": comp_scores["test/f1_hold_shift"],
        "f1_predict_shift": comp_scores["test/f1_predict_shift"],
        "f1_short_long": comp_scores["test/f1_short_long"],
        "f1_bc_prediction": comp_scores["test/f1_bc_prediction"],
        "threshold_pred_shift": [],
        "threshold_short_long": [],
        "threshold_bc_pred": [],
        "model_name": "comparative",
        "predictions_path": [],
        "curve_path": [],
    }

    torch.save(all_score, "assets/score/all_score_new_with_comp.pt")

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


def get_majority_class_probs(events):
    """majority score"""
    # P: next speaker (used on active as well)
    # bc: probability of bc/short/long
    p = torch.zeros_like(events["shift"])
    bc_probs = torch.zeros_like(events["short"])

    # We want to guess all holds
    wh = torch.where(events["hold"])
    p[wh] = torch.ones(len(wh[0]), device=p.device, dtype=torch.float)
    wh = torch.where(events["shift"])
    p[wh] = torch.zeros(len(wh[0]), device=p.device, dtype=torch.float)

    # predict shift -> always predict hold
    w = torch.where(events["predict_shift_pos"])
    p[w] = torch.zeros(len(w[0]), device=p.device, dtype=torch.float)
    w = torch.where(events["predict_shift_neg"])
    p[w] = torch.ones(len(w[0]), device=p.device, dtype=torch.float)

    # BC / Short vs long

    # always predict short
    w = torch.where(events["short"])
    bc_probs[w] = torch.ones(len(w[0]), device=bc_probs.device, dtype=torch.float)
    w = torch.where(events["long"])
    bc_probs[w] = torch.ones(len(w[0]), device=bc_probs.device, dtype=torch.float)

    # Predict bc: Never predict bc
    w = torch.where(events["predict_bc_pos"])
    # bc_probs[w] = 0.
    bc_probs[w] = torch.zeros(len(w[0]), device=bc_probs.device, dtype=torch.float)
    w = torch.where(events["predict_bc_neg"])
    # bc_probs[w] = 0.
    bc_probs[w] = torch.zeros(len(w[0]), device=bc_probs.device, dtype=torch.float)
    return {"p": p, "bc_prediction": bc_probs}


def majority_class_eval():
    everything_deterministic()

    # Load data (same across folds)
    frame_hz = 100
    dm = load_dm(vad_hz=frame_hz, horizon=2, batch_size=16, num_workers=4)

    max_batch = 1000
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
    print(r)

    # Majority class (S/H: all hold, S/L: all short, s-pred: only hold prediction, bc_pred: no bc prediction)
    # {'f1_hold_shift': tensor(0.8430, device='cuda:0'),
    #  'f1_predict_shift': tensor(0.3333, device='cuda:0'),
    #  'f1_short_long': tensor(0.5651, device='cuda:0'),
    #  'f1_bc_prediction': tensor(0.3333, device='cuda:0'),
    #  'shift': {'f1': tensor(0.0005, device='cuda:0'),
    #   'precision': tensor(1., device='cuda:0'),
    #   'recall': tensor(0.0002, device='cuda:0'),
    #   'support': tensor(40957, device='cuda:0')},
    #  'hold': {'f1': tensor(0.9437, device='cuda:0'),
    #   'precision': tensor(0.8933, device='cuda:0'),
    #   'recall': tensor(1., device='cuda:0'),
    #   'support': tensor(342935, device='cuda:0')}}

    from torchmetrics import F1Score

    f1s = 0
    f1h = 0.9437
    ns = 40957
    nh = 342935
    f1w = nh * f1h / (ns + nh)

    # label = torch.cat((torch.ones((ns,)), torch.zeros((nh,)))).long()
    # pred = torch.zeros_like(label, dtype=torch.float)
    label = torch.cat((torch.zeros((ns,)), torch.ones((nh,)))).long()
    pred = torch.ones_like(label, dtype=torch.float)
    #########################################################
    sh_f1 = F1Score(threshold=0.5, num_classes=2, multiclass=True, average="weighted")
    # sh_f1 = F1Score(threshold=0.5)
    sh_f1.update(preds=pred, target=label)
    sh = sh_f1.compute()
    print(sh)


def ANOVA():
    """statistics over aggreagate results"""
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


if __name__ == "__main__":

    everything_deterministic()

    majority_class_eval()
