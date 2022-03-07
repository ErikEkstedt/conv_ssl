from os.path import join
from tqdm import tqdm

import torch
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

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
        result = test(model, dm, online=False)
        all_data[kfold] = result
        # add results
        for metric, value in result[0].items():
            if metric not in all_result:
                all_result[metric] = []
            all_result[metric].append(value)

    return all_result, all_data


def data_ready():
    all_result = torch.load("all_result_discrete.pt")
    all_data = torch.load("all_data_discrete.pt")

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


if __name__ == "__main__":

    # run is specified by <entity>/<project>/<run_id>
    project_id = "how_so/VPModelDiscrete"

    # metrics = ['loss_vp', 'val_loss', 'val_f1_weighted', 'val_f1_pre_weighted', 'val_f1_pw', 'val_bc_prediction']
    metrics = [
        "val_f1_weighted",
        "val_f1_pre_weighted",
        "val_f1_pw",
        "val_bc_ongoing",
        "val_bc_prediction",
    ]
    # data = extract_metrics(discrete, metrics=metrics)

    data = extract_metrics(independent, metrics=metrics, project_id="how_so/VPModel")

    n_fig = len(metrics)
    fig, ax = plt.subplots(n_fig, 1, sharex=True)
    for kfold, tmp_data in data.items():
        for i, (metric, (x, y)) in enumerate(tmp_data.items()):
            ax[i].plot(x, y)
            # ax[i].plot(x, y, label=metric)
            # ax[i].legend()
    for i, _ in enumerate(tmp_data.items()):
        ax[i].set_ylabel(metrics[i])
        ax[i].set_xlabel("step")
    plt.show()

    # all_result, all_data = test_models(discrete)
    # torch.save(all_result, "all_result_discrete.pt")
    # torch.save(all_data, "all_data_discrete.pt")

    all_result, all_data = test_models(independent, project_id="how_so/VPModel")
    torch.save(all_result, "all_result_independent.pt")
    torch.save(all_data, "all_data_independent.pt")
