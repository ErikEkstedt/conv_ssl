import matplotlib.pyplot as plt
import torch

from datasets_turntaking.features.plot_utils import plot_melspectrogram


def __envelope(x, hop):
    """Compute the max-envelope of non-overlapping frames of x at length hop

    x is assumed to be multi-channel, of shape (n_channels, n_samples).
    """
    x_frame = x.unfold(-1, size=hop, step=hop).abs()
    return x_frame.max(dim=-1).values


def plot_waveform(waveform, ax, hop_time, sample_rate):
    # Waveform
    hop_samples = int(hop_time * sample_rate)
    e = __envelope(waveform, hop_samples)

    where = torch.arange(e.shape[-1])
    ax.fill_between(where, y1=-e, y2=e)
    ax.set_xlim([0, len(e)])
    ax.set_ylim([-1.0, 1.0])
    ax.set_ylabel("waveform")
    return ax


def plot_melspectrogram(
    waveform, ax, n_mels=80, frame_time=0.05, hop_time=0.01, sample_rate=16000
):
    waveform = waveform.detach().cpu()

    # Features
    frame_length = int(frame_time * sample_rate)
    hop_length = int(hop_time * sample_rate)
    melspec = Sequential(
        AT.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=frame_length,
            hop_length=hop_length,
            n_mels=n_mels,
        ),
        AT.AmplitudeToDB(),
    )(waveform)

    # im = ax.imshow(melspec, aspect="auto", interpolation="none", origin="lower")
    im = ax.imshow(
        melspec,
        aspect="auto",
        interpolation="none",
        origin="lower",
        extent=(0, melspec.shape[1], 0, melspec.shape[0]),
    )
    return melspec


def plot_vad_oh(
    vad_oh,
    ax=None,
    colors=["b", "orange"],
    yticks=["B", "A"],
    ylabel=None,
    alpha=1,
    label=(None, None),
    legend_loc="best",
    plot=False,
):
    """
    vad_oh:     torch.Tensor: (N, 2)
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    x = torch.arange(vad_oh.shape[0]) + 0.5  # fill_between step = 'mid'
    ax.fill_between(
        x,
        y1=0,
        y2=vad_oh[:, 0],
        step="mid",
        alpha=alpha,
        color=colors[0],
        label=label[1],
    )
    ax.fill_between(
        x,
        y1=0,
        y2=-vad_oh[:, 1],
        step="mid",
        alpha=alpha,
        label=label[0],
        color=colors[1],
    )
    if label[0] is not None:
        ax.legend(loc=legend_loc)
    ax.hlines(y=0, xmin=0, xmax=len(x), color="k", linestyle="dashed")
    ax.set_xlim([0, vad_oh.shape[0]])
    ax.set_ylim([-1.05, 1.05])

    if yticks is None:
        ax.set_yticks([])
    else:
        ax.set_yticks([-0.5, 0.5])
        ax.set_yticklabels(yticks)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    if plot:
        plt.pause(0.1)
    return fig, ax


def plot_static(
    waveform,
    vad,
    audio_kwargs=dict(sample_rate=16000, frame_time=0.05, hop_time=0.01),
    fig_kwargs=dict(
        figsize=(16, 9),
        dpi=100,
        layout=dict(
            left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.02, hspace=0.02
        ),
    ),
    plot=False,
):
    """
    Arguments:
        waveform:       torch.Tensor, (n_samples,)
        vad:            torch.Tensor, (n_frames, 2)
    """
    fig, ax = plt.subplots(4, 1, figsize=fig_kwargs["figsize"], dpi=fig_kwargs["dpi"])

    assert len(ax) >= 4, "Must provide at least 4 ax"

    # MelSpectrogram
    _ = plot_melspectrogram(
        waveform,
        ax=ax[0],
        n_mels=80,
        frame_time=audio_kwargs["frame_time"],
        hop_time=audio_kwargs["hop_time"],
        sample_rate=audio_kwargs["sample_rate"],
    )
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    ax[0].set_ylabel("mel")

    _ = plot_waveform(
        waveform, ax, hop_time=0.01, sample_rate=audio_kwargs["sample_rate"]
    )
    ax[1].set_yticks([])
    ax[1].set_xticks([])

    # VAD
    _ = plot_vad_oh(
        vad.permute(1, 0),
        ax=ax[2],
        label=["A", "B"],
        legend_loc="upper right",
        plot=False,
    )
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_ylabel("vad")

    _ = plot_vad_oh(vad.permute(1, 0), ax=ax[3], alpha=0.1, plot=False)
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].set_ylabel("vad pred")

    plt.tight_layout()
    plt.subplots_adjust(**fig_kwargs["layout"])

    if plot:
        plt.pause(0.1)

    return fig, ax


def plot_entropy(entropy, ax, label="entropy", ylim=[0, 1], cutoff=0.4, color="g"):
    ax_ent = ax.twinx()
    n = len(entropy)
    ent_over_cutoff = entropy >= cutoff
    ax_ent.plot(entropy, label=label, color=color, linewidth=3)
    ax_ent.hlines(
        y=cutoff,
        xmin=0,
        xmax=n,
        linestyle="dashed",
        color=color,
        label="cutoff",
    )
    ax_ent.fill_between(
        torch.arange(n),
        y1=torch.zeros(n),
        y2=torch.ones(n),
        where=ent_over_cutoff,
        color=color,
        alpha=0.07,
    )
    if ylim is not None:
        ax_ent.set_ylim(ylim)
    ax_ent.legend()


def plot_probs(probs, ax, label="shift %", color="r"):
    ax_ts = ax.twinx()
    ax_ts.plot(probs, label=label, color=color, linewidth=2)
    ax_ts.legend(loc="upper left")
    ax_ts.set_ylim([0, 1])
