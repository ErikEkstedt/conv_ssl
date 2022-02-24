import matplotlib.pyplot as plt
import torch
from torch.nn import Sequential
import torchaudio.transforms as AT
import numpy as np

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


def plot_area(oh, ax, y1=-1, y2=1, label=None, color="b", alpha=1, **kwargs):
    ax.fill_between(
        torch.arange(oh.shape[0]),
        y1=y1,
        y2=y2,
        where=oh,
        color=color,
        alpha=alpha,
        label=label,
        **kwargs
    )


def plot_vp_head(
    sil_probs,
    act_probs,
    vad,
    valid,
    hold,
    shift,
    area_alpha=0.3,
    min_context_frames=100,
    horizon=100,
    plot=True,
):
    sil_probs = sil_probs.cpu()
    act_probs = act_probs.cpu()
    vad = vad.cpu()
    valid = valid.cpu()
    hold = hold.cpu()
    shift = shift.cpu()
    valid_shift = torch.logical_and(valid.unsqueeze(-1), shift.cpu())
    valid_hold = torch.logical_and(valid.unsqueeze(-1), hold.cpu())

    N = vad.shape[0]
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(16, 9))
    ##############################################################################3
    # Next speaker A
    n_fig = 0
    ax[n_fig].plot(sil_probs[:, 0], label="Silence A next", color="b")
    ax[n_fig].plot(
        act_probs[:, 0],
        label="Active A next",
        color="darkblue",
        linestyle="dotted",
    )
    _ = plot_vad_oh(vad.cpu(), ax=ax[n_fig].twinx(), alpha=0.6)
    n_fig += 1
    ##############################################################################3
    # Next speaker B
    ax[n_fig].plot(sil_probs[:, 1], label="Silence B next", color="orange")
    ax[n_fig].plot(
        act_probs[:, 1],
        label="Active B next",
        color="darkorange",
        linestyle="dotted",
    )
    _ = plot_vad_oh(vad.cpu(), ax=ax[n_fig].twinx(), alpha=0.6)
    n_fig += 1
    ##############################################################################3
    # VALID
    _ = plot_vad_oh(vad.cpu(), ax=ax[n_fig].twinx(), alpha=0.6)
    plot_area(valid, ax=ax[n_fig], label="VALID", color="k", alpha=area_alpha)
    ax[n_fig].vlines(
        x=[min_context_frames, N - horizon],
        ymin=-1,
        ymax=1,
        color="k",
        linestyle="dashed",
        linewidth=3,
    )
    n_fig += 1
    ##############################################################################3
    # VALID Hold/Shift
    _ = plot_vad_oh(vad.cpu(), ax=ax[n_fig].twinx(), alpha=0.6)
    plot_area(
        valid_shift[:, 0], ax=ax[n_fig], label="Shift", color="g", alpha=area_alpha
    )
    plot_area(valid_shift[:, 1], ax=ax[n_fig], color="g", alpha=area_alpha)
    plot_area(valid_hold[:, 0], ax=ax[n_fig], label="Hold", color="r", alpha=area_alpha)
    plot_area(valid_hold[:, 1], ax=ax[n_fig], color="r", alpha=area_alpha)
    ax[n_fig].vlines(
        x=[min_context_frames, N - horizon],
        ymin=-1,
        ymax=1,
        color="k",
        linestyle="dashed",
        linewidth=3,
    )
    n_fig += 1
    for a in ax:
        a.legend(loc="upper left", fontsize=12)
    if plot:
        plt.pause(0.1)
    return fig, ax


def plot_vad_label(
    vad_label_oh,
    frames=[10, 20, 30, 40],
    colors=["b", "orange"],
    yticks=["B", "A"],
    ylabel=None,
    label=(None, None),
    legend_loc="best",
    alpha=0.9,
    ax=None,
    figsize=(6, 4),
    plot=False,
):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    frame_starts = torch.tensor(frames).cumsum(0)[:-1]
    x = torch.arange(sum(frames))
    expanded = []
    for i, f in enumerate(frames):
        expanded.append(vad_label_oh[:, i].repeat(f, 1))
    expanded = torch.cat(expanded)

    if expanded[:, 0].sum() > 0:
        ax.fill_between(
            x,
            y1=0,
            y2=expanded[:, 0],
            step="pre",
            alpha=alpha,
            color=colors[0],
            label=label[0],
        )
    if expanded[:, 1].sum() > 0:
        ax.fill_between(
            x,
            y1=-expanded[:, 1],
            y2=0,
            step="pre",
            alpha=alpha,
            color=colors[1],
            label=label[1],
        )
    ax.set_ylim([-1.05, 1.05])
    ax.set_xlim([0, len(x) - 1])
    ax.set_xticks([])
    ax.hlines(y=0, xmin=0, xmax=len(x), color="k", linestyle="dashed", linewidth=1)
    ax.vlines(x=frame_starts - 1, ymin=-1, ymax=1, color="k", linewidth=2)

    if label[0] is not None:
        ax.legend(loc=legend_loc)

    if yticks is None:
        ax.set_yticks([])
    else:
        ax.set_yticks([-0.5, 0.5])
        ax.set_yticklabels(yticks)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if plot:
        plt.tight_layout()
        plt.pause(0.01)
    return fig, ax


def plot_labels(label_oh, n_rows, n_cols, figtitle=None, plot=True):
    j = 0
    fig, ax = plt.subplots(
        n_rows, n_cols, sharex=True, figsize=(4 * n_cols, 2 * n_rows)
    )
    for row in range(n_rows):
        for col in range(n_cols):
            plot_vad_label(label_oh[j], ax=ax[row, col])
            j += 1
            if j >= label_oh.shape[0]:
                break
        if j >= label_oh.shape[0]:
            break

    if figtitle is not None:
        fig.suptitle(figtitle, fontsize=15, fontweight="bold")

    plt.tight_layout()
    if plot:
        plt.pause(0.1)
    return fig, ax


def plot_all_labels(vad_projection_head, next_speaker):
    on_silent_shift_oh = vad_projection_head.idx_to_onehot(
        vad_projection_head.on_silent_shift
    )
    on_silent_hold_oh = vad_projection_head.idx_to_onehot(
        vad_projection_head.on_silent_hold
    )
    on_active_shift_oh = vad_projection_head.idx_to_onehot(
        vad_projection_head.on_active_shift
    )
    on_active_hold_oh = vad_projection_head.idx_to_onehot(
        vad_projection_head.on_active_hold
    )
    print("on_silent_shift_oh: ", tuple(on_silent_shift_oh.shape))
    print("on_silent_hold_oh: ", tuple(on_silent_hold_oh.shape))
    print("on_active_shift_oh: ", tuple(on_active_shift_oh.shape))
    print("on_active_hold_oh: ", tuple(on_active_hold_oh.shape))
    ssh = plot_labels(
        on_silent_shift_oh[next_speaker], n_rows=2, n_cols=2, figtitle="SILENT SHIFT"
    )
    sho = plot_labels(
        on_silent_hold_oh[next_speaker], n_rows=2, n_cols=2, figtitle="SILENT HOLD"
    )
    ash = plot_labels(
        on_active_shift_oh[next_speaker], n_rows=3, n_cols=4, figtitle="ACTIVE SHIFT"
    )
    aho = plot_labels(
        on_active_hold_oh[next_speaker], n_rows=2, n_cols=2, figtitle="ACTIVE HOLD"
    )


def plot_next_speaker_probs(
    p_next, vad, shift_prob=None, shift=None, hold=None, plot=True
):
    a_prob = p_next[:, 0].cpu()
    b_prob = p_next[:, 1].cpu()
    v = vad.cpu()

    n = 3
    if shift_prob is not None:
        n = 4
        shift_prob = shift_prob.cpu()
    fig, ax = plt.subplots(n, 1, sharex=True, figsize=(9, 6))
    ##############################################################################3
    # Next speaker A
    n_fig = 0
    twin = ax[n_fig].twinx()
    _ = plot_vad_oh(v, ax=twin, alpha=0.6)

    if shift is not None:
        plot_area(shift, ax=twin, label="Shift", color="g", alpha=0.2)
    if hold is not None:
        plot_area(hold, ax=twin, label="Shift", color="r", alpha=0.2)
    twin.legend(loc="upper right")
    ax[n_fig].plot(a_prob, label="A next speaker", color="b", linewidth=2)
    ax[n_fig].set_ylim([0, 1])

    n_fig += 1
    ##############################################################################3
    # Next speaker B
    twin = ax[n_fig].twinx()
    _ = plot_vad_oh(v, ax=twin, alpha=0.6)
    if shift is not None:
        plot_area(shift, ax=twin, label="Shift", color="g", alpha=0.2)
    if hold is not None:
        plot_area(hold, ax=twin, label="Shift", color="r", alpha=0.2)
    twin.legend(loc="upper right")
    ax[n_fig].plot(b_prob, label="B next speaker", color="orange", linewidth=2)
    ax[n_fig].set_ylim([0, 1])
    n_fig += 1
    if shift_prob is not None:
        twin = ax[n_fig].twinx()
        _ = plot_vad_oh(v, ax=twin, alpha=0.6)
        if shift is not None:
            plot_area(shift, ax=twin, label="Shift", color="g", alpha=0.2)
        if hold is not None:
            plot_area(hold, ax=twin, label="Shift", color="r", alpha=0.2)
        twin.legend(loc="upper right")
        ax[n_fig].plot(shift_prob, label="Shift", color="g", linewidth=2)
        ax[n_fig].set_ylim([0, 1])
        n_fig += 1
    ##############################################################################3
    # DIFF
    diff = a_prob - b_prob
    ax[n_fig].hlines(y=0, xmin=0, xmax=diff.shape[0], color="k", linewidth=2)
    ax[n_fig].fill_between(
        torch.arange(diff.shape[0]),
        y1=0,
        y2=diff,
        where=diff > 0,
        color="b",
    )
    ax[n_fig].fill_between(
        torch.arange(diff.shape[0]),
        y1=diff,
        y2=0,
        where=diff < 0,
        color="orange",
    )
    if shift is not None:
        plot_area(shift, ax=ax[n_fig], label="Shift", color="g", alpha=0.2)
    if hold is not None:
        plot_area(hold, ax=ax[n_fig], label="Shift", color="r", alpha=0.2)
    ax[n_fig].set_ylim([-1, 1])

    for a in ax:
        a.legend(loc="upper left")
    ax[-1].set_xlim([0, a_prob.shape[0]])
    if plot:
        plt.pause(0.1)
    return fig, ax


def plot_window(
    probs,
    vad,
    hold,
    shift,
    pre_hold,
    pre_shift,
    backchannels,
    only_over_05=True,
    plot_kwargs=dict(
        alpha_event=0.2,
        alpha_vad=0.6,
        alpha_probs=1.0,
        shift_hatch=".",
        shift_pre_hatch=".",
        hold_hatch="/",
        hold_pre_hatch="/",
        bc_hatch="x",
        alpha_bc=0.2,
        linewidth=2,
    ),
    ax=None,
    figsize=(12, 6),
    plot=False,
):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax = [ax]

    # cpu
    probs = probs.detach().cpu()
    vad = vad.detach().cpu()
    hold = hold.detach().cpu()
    shift = shift.detach().cpu()
    pre_hold = pre_hold.detach().cpu()
    pre_shift = pre_shift.detach().cpu()
    backchannels = backchannels.detach().cpu()

    n = 0
    # Events
    _ = plot_vad_oh(
        vad,
        ax=ax[n],
        alpha=plot_kwargs["alpha_vad"],
        legend_loc="upper right",
        label=["B", "A"],
    )

    # Backchannels
    if backchannels[..., 0].sum() > 0:
        plot_area(
            backchannels[..., 0],
            color="b",
            alpha=plot_kwargs["alpha_bc"],
            hatch=plot_kwargs["bc_hatch"],
            y1=-0.5,
            y2=0,
            ax=ax[n],
        )
    if backchannels[..., 1].sum() > 0:
        plot_area(
            backchannels[..., 1],
            color="orange",
            alpha=plot_kwargs["alpha_bc"],
            hatch=plot_kwargs["bc_hatch"],
            y1=0,
            y2=0.5,
            ax=ax[n],
        )

    # HOLD
    if hold[:, 0].sum() > 0:
        plot_area(
            hold[:, 0],
            color="b",
            alpha=plot_kwargs["alpha_event"],
            ax=ax[n],
            hatch=plot_kwargs["hold_hatch"],
            label="hold",
        )
    if hold[:, 1].sum() > 0:
        plot_area(
            hold[:, 1],
            color="orange",
            alpha=plot_kwargs["alpha_event"],
            ax=ax[n],
            hatch=plot_kwargs["hold_hatch"],
            label="hold",
        )
    # PRE
    if pre_hold[:, 0].sum() > 0:
        plot_area(
            pre_hold[:, 0],
            color="b",
            alpha=plot_kwargs["alpha_event"],
            ax=ax[n],
            hatch=plot_kwargs["hold_pre_hatch"],
        )
    if pre_hold[:, 1].sum() > 0:
        plot_area(
            pre_hold[:, 1],
            color="orange",
            alpha=plot_kwargs["alpha_event"],
            ax=ax[n],
            hatch=plot_kwargs["hold_pre_hatch"],
        )

    # HOLD
    if shift[:, 0].sum() > 0:
        plot_area(
            shift[:, 0],
            color="b",
            alpha=plot_kwargs["alpha_event"],
            ax=ax[n],
            hatch=plot_kwargs["shift_hatch"],
            label="shift",
        )
    if shift[:, 1].sum() > 0:
        plot_area(
            shift[:, 1],
            color="orange",
            alpha=plot_kwargs["alpha_event"],
            ax=ax[n],
            hatch=plot_kwargs["shift_hatch"],
            label="shift",
        )
    if pre_shift[:, 0].sum() > 0:
        plot_area(
            pre_shift[:, 0],
            color="b",
            alpha=plot_kwargs["alpha_event"],
            ax=ax[n],
            hatch=plot_kwargs["shift_pre_hatch"],
        )
    if pre_shift[:, 1].sum() > 0:
        plot_area(
            pre_shift[:, 1],
            color="orange",
            alpha=plot_kwargs["alpha_event"],
            ax=ax[n],
            hatch=plot_kwargs["shift_pre_hatch"],
        )

    # Speaker Probs
    axes = ax[n].twinx()

    if only_over_05:
        wa = torch.where(probs[:, 0] >= 0.5)[0]
        wb = torch.where(probs[:, 1] >= 0.5)[0]
        pa = np.zeros_like(probs[:, 0])
        pb = np.zeros_like(probs[:, 0])
        pa[wa] = probs[:, 0][wa]
        pa[wb] = np.nan
        pb[wb] = probs[:, 1][wb]
        pb[wa] = np.nan
    else:
        pa = probs[:, 0]
        pb = probs[:, 1]

    axes.plot(
        pa,
        color="b",
        linewidth=plot_kwargs["linewidth"],
        label="A is next",
        alpha=plot_kwargs["alpha_probs"],
    )
    axes.plot(
        pb,
        color="orange",
        linewidth=plot_kwargs["linewidth"],
        label="B is next",
        alpha=plot_kwargs["alpha_probs"],
    )
    axes.set_ylim([-0.05, 1.05])
    axes.set_yticks([])

    axes.legend(loc="upper left")
    ax[n].legend(loc="lower left")

    if plot:
        plt.tight_layout()
        plt.pause(0.001)

    return fig, ax
