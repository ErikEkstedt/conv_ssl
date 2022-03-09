from os.path import join
from tqdm import tqdm

import torch
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import pytorch_lightning as pl

from conv_ssl.utils import everything_deterministic
from conv_ssl.evaluation.utils import load_model, load_dm
from conv_ssl.evaluation.evaluation import test


api = wandb.Api()

"""
* Download all kfold metrics
    - to show training plots
* Organize all artifacts to evaluate test
"""


discrete = {
    "0": None,
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
    "0": None,
    "1": None,
    "2": None,
    "3": None,
    "4": None,
    "5": None,
    "6": None,
    "7": None,
    "8": None,
    "9": None,
    "10": None,
}

comparative = {
    "0": None,
    "1": None,
    "2": None,
    "3": None,
    "4": None,
    "5": None,
    "6": None,
    "7": None,
    "8": None,
    "9": None,
    "10": None,
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


def test_models(id_dict, project_id="how_so/VPModel"):
    project_id = "how_so/VPModel"

    # update vad_projection metrics
    metric_kwargs = {
        "event_pre": 0.5,
        "event_min_context": 1.0,
        "event_min_duration": 0.15,
        "event_horizon": 1.0,
        "event_start_pad": 0.05,
        "event_target_duration": 0.10,
        "event_bc_pre_silence": 1,
        "event_bc_post_silence": 2,
        "event_bc_max_active": 1,
        "event_bc_prediction_window": 0.5,
    }

    all_data = {}
    all_result = {}
    for kfold, id in id_dict.items():
        if id is None:
            continue
        run_path = join(project_id, id)
        print(f"{kfold} run_path: ", run_path)
        # Load model and process test-set
        # Load data (same across folds)
        dm = load_dm(vad_hz=100, horizon=2, batch_size=16, num_workers=4)
        model = load_model(run_path=run_path, eval=True)
        # metric_kwargs metrics
        for metric, val in metric_kwargs.items():
            model.conf["vad_projection"][metric] = val
        model.test_metric = model.init_metric(model.conf, model.frame_hz)
        model.test_metric = model.test_metric.to(model.device)
        result = test(model, dm, online=False)
        all_data[kfold] = result
        # add results
        for metric, value in result[0].items():
            if metric not in all_result:
                all_result[metric] = []
            all_result[metric].append(value)

    return all_result, all_data


def plot_result_histogram(fig_data, plot=True):
    off = -1.5
    pad = 0.30
    w = pad - 0.02
    xx = torch.arange(len(metrics))
    colors = ["b", "orange", "red", "green"]
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
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
    ax.set_ylim([0.77, 0.9])
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
    }

    metrics = ["f1_weighted", "f1_pre_weighted", "bc_ongoing"]
    fig_data = {}
    for model, results in data.items():
        fig_data[model] = {}
        for metric in metrics:
            x = torch.tensor(data[model][metric])
            fig_data[model][metric] = {"mean": x.mean(), "std": x.std()}

    fig, ax = plot_result_histogram(fig_data, plot=True)

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
    dm = load_dm(vad_hz=100, horizon=2, batch_size=4, num_workers=4)
    model = load_model(run_path=run_path, eval=True)
    metric_kwargs = {
        "event_pre": 0.5,
        "event_min_context": 1.0,  # min context
        "event_horizon": 1.0,  # lookahead after shift/holds
        "event_min_duration": 0.15,
        "event_start_pad": 0.05,
        "event_target_duration": 0.10,
        "event_bc_pre_silence": 1,
        "event_bc_post_silence": 2,
        "event_bc_max_active": 1,
        "event_bc_prediction_window": 0.5,
    }
    for metric, val in metric_kwargs.items():
        model.conf["vad_projection"][metric] = val
    model.test_metric = model.init_metric(model.conf, model.frame_hz)
    model.test_metric = model.test_metric.to(model.device)

    max_batches = 1000
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

    # run is specified by <entity>/<project>/<run_id>
    # project_id = "how_so/VPModelDiscrete"
    # metrics = ['loss_vp', 'val_loss', 'val_f1_weighted', 'val_f1_pre_weighted', 'val_f1_pw', 'val_bc_prediction']
    # metrics = [
    #     "val_f1_weighted",
    #     "val_f1_pre_weighted",
    #     "val_f1_pw",
    #     "val_bc_ongoing",
    #     "val_bc_prediction",
    # ]
    # data = extract_metrics(discrete, metrics=metrics)
    # data = extract_metrics(independent, metrics=metrics, project_id="how_so/VPModel")
    # n_fig = len(metrics)
    # fig, ax = plt.subplots(n_fig, 1, sharex=True)
    # for kfold, tmp_data in data.items():
    #     for i, (metric, (x, y)) in enumerate(tmp_data.items()):
    #         ax[i].plot(x, y)
    #         # ax[i].plot(x, y, label=metric)
    #         # ax[i].legend()
    # for i, _ in enumerate(tmp_data.items()):
    #     ax[i].set_ylabel(metrics[i])
    #     ax[i].set_xlabel("step")
    # plt.show()

    pl.seed_everything(100)
    everything_deterministic()

    all_result, all_data = test_models(discrete, project_id="how_so/VPModel")
    torch.save(all_result, "all_result_discrete.pt")
    torch.save(all_data, "all_data_discrete.pt")

    all_result, all_data = test_models(independent, project_id="how_so/VPModel")
    torch.save(all_result, "all_result_independent.pt")
    torch.save(all_data, "all_data_independent.pt")

    all_result, all_data = test_models(
        independent_baseline, project_id="how_so/VPModel"
    )
    torch.save(all_result, "all_result_ind_base.pt")
    torch.save(all_data, "all_data_ind_base.pt")

    all_result, all_data = test_models(comparative, project_id="how_so/VPModel")
    torch.save(all_result, "all_result_comp.pt")
    torch.save(all_data, "all_data_comp.pt")
