import torch

from copy import deepcopy
from tqdm import tqdm

from conv_ssl.model import VPModel
from conv_ssl.augmentations import flatten_pitch_batch
from conv_ssl.evaluation.phrase_dataset import PhraseDataset
from conv_ssl.plot_utils import plot_vad_oh

import matplotlib.pyplot as plt


def plot_next_speaker(p_a, ax, color=["b", "orange"], alpha=0.6):
    p_ns = p_a
    x = torch.arange(len(p_a))
    ax.fill_between(
        x,
        y1=0.5,
        y2=p_ns,
        where=p_ns > 0.5,
        alpha=alpha,
        color=color[0],
        label="A turn",
    )
    ax.fill_between(
        x,
        y1=p_ns,
        y2=0.5,
        where=p_ns < 0.5,
        alpha=alpha,
        color=color[1],
        label="B turn",
    )
    return ax


def plot_bc(p_bc, ax, color=["g", "g"], alpha=0.4):
    x = torch.arange(p_bc.shape[0])
    ax.fill_between(
        x, y1=0, y2=p_bc[:, 0], alpha=alpha, color=color[0], label="BC/SHORT A"
    )
    ax.fill_between(
        x, y1=-1, y2=p_bc[:, 1] - 1, alpha=alpha, color=color[1], label="BC/SHORT B"
    )


def plot_batch(probs, batch, frame_hz=100, rows=5, cols=2, fontsize=15):
    nb, _, _ = batch["vad"].shape

    N = 2
    fig, ax = plt.subplots(nb * N, 1, sharex=True, figsize=(12, 12))

    for i in range(nb):
        ii = i * N
        plot_vad_oh(batch["vad"][i], ax=ax[ii], alpha=1, colors=["blue", "red"])
        _ = plot_next_speaker(
            p_a=probs["p"][i, :, 0],
            ax=ax[ii + 1],
            alpha=0.6,
            color=["cornflowerblue", "pink"],
        )
        ax[ii + 1].legend(loc="upper left")
        ax[ii + 1].set_yticks([0, 0.5, 1])

        if "starts" in batch and "words" in batch:
            for word, start in zip(batch["words"][i], batch["starts"][i]):
                frame_start = int(start * frame_hz)
                ax[ii].text(
                    x=frame_start,
                    y=0,
                    s=word,
                    fontsize=fontsize,
                    horizontalalignment="left",
                )
                ax[ii + 1].text(
                    x=frame_start,
                    y=0.5,
                    s=word,
                    fontsize=fontsize,
                    horizontalalignment="left",
                )

    plt.tight_layout()
    plt.pause(0.1)
    # plt.show()
    return fig, ax


def plot_single(probs, batch, n_batch, frame_hz, suptitle=None, fontsize=15):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    if suptitle is not None:
        fig.suptitle(suptitle)
    plot_vad_oh(batch["vad"][n_batch], ax=ax[0], alpha=1, colors=["blue", "red"])
    _ = plot_next_speaker(
        p_a=probs["p"][n_batch, :, 0],
        ax=ax[1],
        alpha=0.6,
        color=["cornflowerblue", "pink"],
    )
    ax[1].legend(loc="upper left")
    ax[1].set_yticks([0, 0.5, 1])

    if "starts" in batch and "words" in batch:
        for word, start in zip(batch["words"][n_batch], batch["starts"][n_batch]):
            frame_start = int(start * frame_hz)
            ax[0].text(
                x=frame_start,
                y=0,
                s=word,
                fontsize=fontsize,
                horizontalalignment="left",
            )
            ax[1].text(
                x=frame_start,
                y=0.5,
                s=word,
                fontsize=fontsize,
                horizontalalignment="left",
            )
    plt.tight_layout()
    return fig, ax


def add_to_dict(input_dict, data_dict, short_long):
    special = ["starts", "words"]
    for k, v in input_dict.items():
        if k in data_dict[short_long]:
            if k in special:
                data_dict[short_long][k].append(v[0])
            else:
                data_dict[short_long][k].append(v)
        else:
            if k in special:
                data_dict[short_long][k] = v
            else:
                data_dict[short_long][k] = [v]
    return data_dict


def cat_dict(input_dict):
    for short_long in ["short", "long"]:
        for k, v in input_dict[short_long].items():
            if isinstance(v[0], torch.Tensor):
                input_dict[short_long][k] = torch.cat(v)
    return input_dict


def create_flat_batch(batch):
    flat_waveform = flatten_pitch_batch(batch["waveform"].unsqueeze(1), batch["vad"])
    flat_batch = {"waveform": flat_waveform}
    for k, v in batch.items():
        if k == "waveform":
            continue
        flat_batch[k] = deepcopy(v)
    return flat_batch


if __name__ == "__main__":

    checkpoint = "assets/PaperB/checkpoints/cpc_48_50hz_15gqq5s5.ckpt"
    model = VPModel.load_from_checkpoint(checkpoint)
    model = model.eval()
    _ = model.to("cuda")

    # phrase_path = "assets/phrases/phrases.json"
    phrase_path = "assets/phrases_beta/phrases.json"
    dset = PhraseDataset(
        phrase_path,
        vad_hz=model.frame_hz,
        sample_rate=model.sample_rate,
        vad_horizon=model.VAP.horizon,
        vad_history=model.conf["data"]["vad_history"],
        vad_history_times=model.conf["data"]["vad_history_times"],
    )

    dset.indices[:20]

    batches = {"short": {}, "long": {}}
    probs = {"short": {}, "long": {}}
    flat_batches = {"short": {}, "long": {}}
    flat_probs = {"short": {}, "long": {}}
    for batch in tqdm(dset):
        flat_batch = create_flat_batch(batch)
        # Forward
        loss, out, prob, batch = model.output(batch)
        flat_loss, flat_out, flat_prob, flat_batch = model.output(flat_batch)
        # contain data
        short_long = batch["size"][0]
        name = f"{batch['session']}_{short_long}"
        fig, ax = plot_single(
            prob,
            batch,
            n_batch=0,
            suptitle=name,
            frame_hz=model.frame_hz,
        )
        fig.savefig(f"assets/PaperB/figs/{name}.png")
        fig, ax = plot_single(
            flat_prob,
            flat_batch,
            n_batch=0,
            suptitle=name + "_flat",
            frame_hz=model.frame_hz,
        )
        fig.savefig(f"assets/PaperB/figs/{name}_flat.png")
        plt.close("all")
        # print(short_long)
        # batches = add_to_dict(batch, batches, short_long)
        # probs = add_to_dict(prob, probs, short_long)
        # # Flatten pitch
        # flat_batches = add_to_dict(flat_batch, flat_batches, short_long)
        # flat_probs = add_to_dict(flat_prob, flat_probs, short_long)
    # batches = cat_dict(batches)
    # probs = cat_dict(probs)
    # flat_batches = cat_dict(flat_batches)
    # flat_probs = cat_dict(flat_probs)
