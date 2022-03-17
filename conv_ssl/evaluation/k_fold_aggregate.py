from os.path import join
from tqdm import tqdm

import torch
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import pytorch_lightning as pl

from conv_ssl.utils import everything_deterministic
from conv_ssl.plot_utils import plot_pr_curve
from conv_ssl.evaluation.utils import load_model, load_dm
from conv_ssl.evaluation.evaluation import test


api = wandb.Api()

"""
* Download all kfold metrics
    - to show training plots
* Organize all artifacts to evaluate test
"""


discrete = {
    "0": "1h52tpnn",
    "1": "3fhjobk0",
    "2": "120k8fdv",
    "3": "1vx0omkd",
    "4": "sbzhz86n",
    "5": "1lyezca0",
    "6": "2vtd1u1n",
    "7": "2ldfo4rg",
    "8": "2ca7uxad",
    "9": "2fsy74rf",
    "10": "3ik6jod6",
}

independent = {
    "0": "1t7vvo0c",
    "1": "24bn5wi6",
    "2": "1u7yzji0",
    "3": "s5unjaj7",
    "4": "10krujrj",
    "5": "2rq33fxr",
    "6": "3uqpk8e1",
    "7": "3mpxa1iy",
    "8": "3ulpo767",
    "9": "3d952gec",
    "10": "2651d3ln",
}

independent_baseline = {
    "0": "2mme28tm",
    "1": "qo9mf26t",
    "2": "2rrdm5ma",
    "3": "mrzizwex",
    "4": "1lximsk1",
    "5": "2wyymo7n",
    "6": "1nze8m3l",
    "7": "1cdhj9yo",
    "8": "kamzjel0",
    "9": "15ze1p0y",
    "10": "2mvwxxar",
}

comparative = {
    "0": "2kwhi1zi",
    "1": "2izpsu6r",
    "2": "23mzxhhd",
    "3": "1lvk73tr",
    "4": "11jlsatj",
    "5": "nxvb62j4",
    "6": "1pglrfbn",
    "7": "1z9qyfh6",
    "8": "1kgiwy2m",
    "9": "1eluv8de",
    "10": "2530040o",
}


def extract_metrics(
    id_dict,
    metrics=[
        "val_f1_weighted",
        "val_f1_pre_weighted",
        "val_f1_pw",
        "val_bc_prediction",
    ],
    project_id="how_so/VPModelDiscrete",
):
    data = {}
    for kfold, id in tqdm(id_dict.items(), desc="Download metrics"):
        if id is None:
            continue
        run = api.run(join(project_id, id))
        df = run.history()
        data[kfold] = {}
        for metric in metrics:
            if metric in df:
                idx = df[metric].notna()
                y = df[metric][idx]
                x = df["_step"][idx]
                data[kfold][metric] = [x, y]
    return data


def test_single_model_OLD(run_path, metric_kwargs, bc_pred_pr_curve=False):
    # Load data (same across folds)
    dm = load_dm(vad_hz=100, horizon=2, batch_size=16, num_workers=4)

    # Load model and process test-set
    model = load_model(run_path=run_path, eval=True, strict=False)

    # Updatemetric_kwargs metrics
    for metric, val in metric_kwargs.items():
        model.conf["vad_projection"][metric] = val

    model.test_metric = model.init_metric(
        model.conf, model.frame_hz, bc_pred_pr_curve=bc_pred_pr_curve
    )
    model.test_metric = model.test_metric.to(model.device)
    result = test(model, dm, online=False)
    return result, model


def test_single_model(run_path, event_kwargs, bc_pred_pr_curve=False, batch_size=16):
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
        model.conf, model.frame_hz, bc_pred_pr_curve=bc_pred_pr_curve, **event_kwargs
    )
    model.test_metric = model.test_metric.to(model.device)
    result = test(model, dm, online=False)
    return result, model


def test_models(id_dict, event_kwargs, project_id="how_so/VPModel"):
    all_data = {}
    all_result = {}
    for kfold, id in id_dict.items():
        if id is None:
            continue
        run_path = join(project_id, id)
        print(f"{kfold} run_path: ", run_path)
        result, _ = test_single_model(run_path, event_kwargs=event_kwargs)
        all_data[kfold] = result
        # add results
        for metric, value in result[0].items():
            if metric not in all_result:
                all_result[metric] = []
            all_result[metric].append(value)

    return all_result, all_data


def plot_result_histogram(fig_data, metrics, ylim=None, figsize=(6, 2), plot=True):
    off = -1.5
    pad = 0.20
    w = pad - 0.02
    xx = torch.arange(len(metrics))
    colors = ["b", "orange", "red", "green"]
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i, (model, result) in enumerate(fig_data.items()):
        for xi, metric in enumerate(metrics):
            x_tmp = xx[xi] + (off * pad)
            if xi == 0:
                label = model
            else:
                label = None
            ax.bar(
                x=x_tmp,
                height=result[metric]["mean"],
                yerr=result[metric]["std"],
                alpha=0.5,
                color=colors[i],
                capsize=5,
                width=w,
                label=label,
            )
        off += 1
    ax.legend(loc="upper right")
    if ylim is not None:
        # ax.set_ylim([0.73, 0.9])
        ax.set_ylim(ylim)
    ax.set_xticks(list(range(len(metrics))))
    ax.set_xticklabels(metrics)
    if plot:
        plt.pause(0.01)
    return fig, ax


def data_ready():
    data = {
        "discrete": {
            k.replace("test/", ""): v
            for k, v in torch.load("all_result_discrete.pt").items()
        },
        "independent": {
            k.replace("test/", ""): v
            for k, v in torch.load("all_result_independent.pt").items()
        },
        "independent_baseline": {
            k.replace("test/", ""): v
            for k, v in torch.load("all_result_ind_base.pt").items()
        },
        "comperative": {
            k.replace("test/", ""): v
            for k, v in torch.load("all_result_comp.pt").items()
        },
    }
    data = {
        "discrete": {
            k.replace("test/", ""): v
            for k, v in torch.load("all_result_discrete_new.pt").items()
        },
        "independent": {
            k.replace("test/", ""): v
            for k, v in torch.load("all_result_independent_new.pt").items()
        },
        "independent_baseline": {
            k.replace("test/", ""): v
            for k, v in torch.load("all_result_ind_base_new.pt").items()
        },
    }
    data = {
        "discrete": {
            k.replace("test/", ""): v
            for k, v in torch.load("all_res_discrete_new.pt").items()
        },
        "independent": {
            k.replace("test/", ""): v
            for k, v in torch.load("all_res_independent.pt").items()
        },
    }

    # metrics = ["f1_weighted", "f1_pre_weighted", "f1_bc_ongoing", "bc_prediction"]
    # metrics = ["f1_weighted", "f1_pre_weighted", "f1_bc_ongoing"]
    metrics = ["f1_hold_shift", "f1_predict_shift", "f1_short_long", "f1_bc_prediction"]
    fig_data = {}
    for model, results in data.items():
        fig_data[model] = {}
        for metric in metrics:
            x = torch.tensor(data[model][metric])
            fig_data[model][metric] = {"mean": x.mean(), "std": x.std(unbiased=True)}

    for model, values in fig_data.items():
        print(model)
        for metric, val in values.items():
            print(metric, round(val["mean"].item(), 4), round(val["std"].item(), 4))

    fig, ax = plot_result_histogram(
        fig_data, metrics=metrics, figsize=(6.5, 3), plot=False
    )
    ax.set_ylabel("F1")
    pad = 0.1
    plt.subplots_adjust(
        left=0.12, bottom=0.15, right=0.99, top=1 - pad, wspace=None, hspace=None
    )
    # plt.pause(0.01)
    plt.show()

    for metric in metrics:
        for model in fig_data.keys():
            m = round(fig_data[model][metric]["mean"].item(), 4)
            s = round(fig_data[model][metric]["std"].item(), 4)
            print(f"{model}: {m} ({s})")
    ax.bar(x=xx, y=fig_data[model][metric]["mean"], yerr=fig_data[model][metric]["std"])
    for metric, val in all_result.items():
        val = torch.tensor(val)
        m = round(val.mean().item(), 3)
        s = round(val.std(unbiased=False).item(), 3)
        print(f"{metric.replace('test/', '')}: ", m, s)

    # Errorbars as std
    x, y, yerr = [], [], []
    for metric in all_result.keys():
        if "support" in metric:
            continue
        if "loss" in metric:
            continue
        if "shift" in metric:
            continue
        if "hold" in metric:
            continue
        m = torch.tensor(all_result[metric])
        u = m.mean()
        s = m.std(unbiased=False)
        x.append(metric.replace("test/", ""))
        y.append(u)
        yerr.append(s)
    y, perm = torch.tensor(y).sort(descending=True)
    y = list(y)
    x = [x[i] for i in perm]
    yerr = [yerr[i] for i in perm]

    # Plot bars w/ std for all metrics comparing models
    pad = 0.30
    w = pad - 0.02
    p = pad // 2
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    # ax.bar(x, y, yerr=yerr, alpha=0.5, capsize=5, width=0.18)
    xx = torch.arange(len(x)) * 2
    ax.bar(xx - (1.5 * pad), y, yerr=yerr, alpha=0.5, capsize=5, width=w)
    ax.bar(xx - (0.5 * pad), y, yerr=yerr, alpha=0.5, capsize=5, width=w)
    # ax.bar(xx, y, yerr=yerr, alpha=0.5, capsize=5, width=w)
    ax.bar(xx + (0.5 * pad), y, yerr=yerr, alpha=0.5, capsize=5, width=w)
    ax.bar(xx + (1.5 * pad), y, yerr=yerr, alpha=0.5, capsize=5, width=w)
    ax.set_ylim([0.35, 1.0])
    ax.set_xticklabels([""] + x)
    ax.legend(["discrete", "independent", "independent40", "comparative"])
    plt.show()


def debugging():
    from conv_ssl.utils import to_device

    project_id = "how_so/VPModel"
    id = "sbzhz86n"  # discrete
    # id = "10krujrj"  # independent
    run_path = join(project_id, id)

    # update vad_projection metrics
    metric_kwargs = {
        "event_pre": 0.5,  # seconds used to estimate PRE-f1-SHIFT/HOLD
        "event_min_context": 1.0,  # min context duration before extracting metrics
        "event_min_duration": 0.15,  # the minimum required segment to extract SHIFT/HOLD (start_pad+target_duration)
        "event_horizon": 1.0,  # SHIFT/HOLD requires lookahead to determine mutual starts etc
        "event_start_pad": 0.05,  # Predict SHIFT/HOLD after this many seconds of silence after last speaker
        "event_target_duration": 0.10,  # duration of segment to extract each SHIFT/HOLD guess
        "event_bc_target_duration": 0.25,  # duration of activity, in a backchannel, to extract BC-ONGOING metrics
        "event_bc_pre_silence": 1,  # du
        "event_bc_post_silence": 2,
        "event_bc_max_active": 1.0,
        "event_bc_prediction_window": 0.4,
        "event_bc_neg_active": 1,
        "event_bc_neg_prefix": 1,
        "event_bc_ongoing_threshold": 0.5,
        "event_bc_pred_threshold": 0.5,  # this should be varied
    }
    run_path = "how_so/VPModel/sbzhz86n"  # discrete
    # run_path ="how_so/VPModel/10krujrj"  # independent

    dm = load_dm(vad_hz=100, horizon=2, batch_size=4, num_workers=4)
    model = load_model(run_path=run_path, eval=True, strict=False)
    for metric, val in metric_kwargs.items():
        model.conf["vad_projection"][metric] = val
    model.test_metric = model.init_metric(
        model.conf, model.frame_hz, bc_pred_pr_curve=True
    )
    model.test_metric = model.test_metric.to(model.device)

    max_batches = 100
    model.test_metric.reset()
    if max_batches > 0:
        pbar = tqdm(enumerate(dm.val_dataloader()), total=max_batches)
    else:
        dloader = dm.val_dataloader()
        pbar = tqdm(enumerate(dloader), total=len(dloader))
    for i, batch in pbar:
        batch = to_device(batch)
        # EVENTS
        events = model.test_metric.extract_events(batch["vad"])
        # FORWARD
        _, out, batch = model.shared_step(batch)
        # get probs for zero-shot
        probs = model.get_next_speaker_probs(out["logits_vp"], batch["vad"])
        # UPDATE METRICS
        model.test_metric.update(
            p=probs["p"],
            pre_probs=probs.get("pre_probs", None),
            pw=probs.get("pw", None),
            bc_pred_probs=probs.get("bc_prediction", None),
            events=events,
        )
        if max_batches > 0 and i == max_batches:
            break
    result = model.test_metric.compute()

    # torch.save(result, "tmp_ind.pt")
    torch.save(result, "tmp_discrete.pt")

    result_ind = torch.load("tmp_ind.pt")

    result_discrete = result


if __name__ == "__main__":

    pl.seed_everything(100)
    everything_deterministic()

    # update vad_projection metrics
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
        metric_onset_dur=0.3,
        metric_pre_label_dur=0.5,
        metric_min_context=1.0,
        bc_max_duration=1.0,
        bc_pre_silence=1.0,
        bc_post_silence=1.0,
    )

    # run_path = "how_so/VPModel/sbzhz86n"  # discrete
    # run_path = "how_so/VPModel/10krujrj"  # independent
    #
    # result, model = test_single_model(
    #     run_path, event_kwargs=event_kwargs, bc_pred_pr_curve=False
    # )

    # Choose a model
    torch.tensor(data["discrete"]["test_loss"]).min(dim=0)  # 2
    torch.tensor(data["discrete"]["f1_hold_shift"]).max(dim=0)  # 5
    torch.tensor(data["discrete"]["f1_predict_shift"]).max(dim=0)  # 9
    torch.tensor(data["discrete"]["f1_short_long"]).max(dim=0)  # 2
    torch.tensor(data["discrete"]["f1_bc_prediction"]).max(dim=0)  # 3

    run_path = discrete["3"]
    result, model = test_single_model(
        run_path, event_kwargs=event_kwargs, bc_pred_pr_curve=True
    )
    res = model.test_metric.compute()

    preds = model.test_metric.bc_pred_pr.preds
    target = model.test_metric.bc_pred_pr.target
    preds = torch.cat(preds)
    target = torch.cat(target)

    n_pos = (target == 1).sum() / len(target)
    n_neg = 1 - n_pos

    fig, ax = plt.subplots(2, 1)
    ax[0].hist(np.unique(preds.numpy().round(5)), bins=50)
    plt.pause(0.1)

    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import PrecisionRecallDisplay

    p_rev = 1 - preds
    t_rev = 1 - target
    p, r, t = precision_recall_curve(y_true=target, probas_pred=preds)
    p1, r1, t1 = precision_recall_curve(y_true=target, probas_pred=p_rev)

    a = PrecisionRecallDisplay.from_predictions(y_true=target, y_pred=preds)
    PrecisionRecallDisplay.from_predictions(y_true=target, y_pred=p_rev, ax=a.ax_)
    plt.show()

    fig, ax = plot_pr_curve(
        torch.from_numpy(p.copy()),
        torch.from_numpy(r.copy()),
        torch.from_numpy(t.copy()),
        model="discrete",
        color_auc_max="b",
        # thresh_min=0.01,
        plot=False,
    )
    _ = plot_pr_curve(
        torch.from_numpy(p1.copy()),
        torch.from_numpy(r1.copy()),
        torch.from_numpy(t1.copy()),
        model="discrete rev",
        color_auc_max="g",
        # thresh_min=0.01,
        ax=ax,
        plot=False,
    )
    ax[0].set_ylim([0, 1.05])
    plt.pause(0.01)

    precision, recall, thresholds = res["pr_curve_bc_pred"]

    print(torch.tensor(data["independent"]["test_loss"]).min(dim=0))  # 2
    print(torch.tensor(data["independent"]["f1_hold_shift"]).max(dim=0))  # 4
    print(torch.tensor(data["independent"]["f1_predict_shift"]).max(dim=0))  # 8
    print(torch.tensor(data["independent"]["f1_short_long"]).max(dim=0))  # 6
    print(torch.tensor(data["independent"]["f1_bc_prediction"]).max(dim=0))  # 10

    run_path = independent["10"]
    result, model = test_single_model(
        run_path, event_kwargs=event_kwargs, bc_pred_pr_curve=True
    )
    res = model.test_metric.compute()
    precision, recall, thresholds = res["pr_curve_bc_pred"]

    ind_res = res

    #
    # # torch.save(R, 'discrete_4.pt')
    #

    ###################################################################
    ###################################################################
    ############### PR-CURVES #########################################
    ###################################################################
    ###################################################################
    # R = torch.load("pr_bc_pred_data.pt")
    # precision, recall, thresholds = R["discrete"]
    # precision, recall, thresholds = R["discrete"]
    fig, ax = plot_pr_curve(
        precision.cpu(),
        recall.cpu(),
        thresholds.cpu(),
        thresh_min=0.01,
        model="Discrete",
        plot=False,
    )
    # precision, recall, thresholds = R["independent"]
    prec, rec, thresh = ind_res["pr_curve_bc_pred"]
    _ = plot_pr_curve(
        prec.cpu(),
        rec.cpu(),
        thresh.cpu(),
        model="Independent",
        color_auc_max="b",
        thresh_min=0.01,
        ax=ax,
        plot=False,
    )
    plt.show()

    #
    # precision, recall, thresholds = R["independent"]
    # print("precision: ", tuple(precision.shape))
    # print("recall: ", tuple(recall.shape))
    # print("thresholds: ", tuple(thresholds.shape))
    # fig, ax = plot_pr_curve(
    #     precision.cpu(),
    #     recall.cpu(),
    #     thresholds.cpu(),
    #     model="Independent",
    #     color_auc_max="b",
    #     thresh_min=0,
    #     plot=False,
    # )
    # plt.show()
    # plt.pause(0.01)

    ###################################################################
    ###################################################################
    ############### All models ########################################
    ###################################################################
    ###################################################################

    # all_result, all_data = test_models(
    #     discrete, event_kwargs=event_kwargs, project_id="how_so/VPModel"
    # )
    # torch.save(all_result, "all_result_discrete_new.pt")
    # torch.save(all_data, "all_data_discrete_new.pt")
    # all_result, all_data = test_models(
    #     independent, event_kwargs=event_kwargs, project_id="how_so/VPModel"
    # )
    # torch.save(all_result, "all_result_independent_new.pt")
    # torch.save(all_data, "all_data_independent_new.pt")
    # all_result, all_data = test_models(
    #     independent_baseline, event_kwargs=event_kwargs, project_id="how_so/VPModel"
    # )
    # torch.save(all_result, "all_result_ind_base_new.pt")
    # torch.save(all_data, "all_data_ind_base_new.pt")
    # all_result, all_data = test_models(
    #     comparative, event_kwargs=event_kwargs, project_id="how_so/VPModel"
    # )
    # torch.save(all_result, "all_result_comp_new.pt")
    # torch.save(all_data, "all_data_comp_new.pt")
