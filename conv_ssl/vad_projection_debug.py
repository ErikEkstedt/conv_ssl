import torch

from conv_ssl.plot_utils import plot_vad_oh
from conv_ssl.utils import find_island_idx_len
from conv_ssl.model import ShiftHoldMetric


from datasets_turntaking.features.vad import (
    DialogEvents,
    VadProjection,
)  # , ProjectionCodebook


def to_device(batch, device="cuda"):
    new_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            new_batch[k] = v.to(device)
        else:
            new_batch[k] = v
    return new_batch


def eot_moments(vad, n_frames):
    b, n = vad.shape[:2]
    eots_oh = torch.zeros((2, b, n))
    for batch in range(b):
        for speaker in [0, 1]:
            s, d, v = find_island_idx_len(vad[batch, :, speaker])
            active = v == 1
            if active.sum() > 0:
                start = s[active]
                dur = d[active]
                end = start + dur
                keep = torch.where(end < n)[0]
                if len(keep) > 0:
                    start = start[keep]
                    dur = dur[keep]
                    end = end[keep]
                    over_dur = (dur - n_frames) > 0
                    new = torch.where(over_dur)[0]
                    if len(new) > 0:
                        start[new] = end[new] - n_frames
                    for s, e in zip(start, end):
                        eots_oh[speaker, batch, s : e + 1] = 1
    return eots_oh


def plot_area(oh, ax, label=None, color="b", alpha=1):
    ax.fill_between(
        torch.arange(oh.shape[0]),
        y1=-1,
        y2=1,
        where=oh,
        color=color,
        alpha=alpha,
        label=label,
    )


def plot_vp_head(
    sil_probs, act_probs, vad, valid, hold, shift, area_alpha=0.3, plot=True
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
    if plot:
        plt.pause(0.1)
    return fig, ax


def explanation():
    vad_projection_head = VadProjection(bin_sizes)
    print("on_silent_shift: ", tuple(vad_projection_head.on_silent_shift.shape))
    print("on_silent_hold: ", tuple(vad_projection_head.on_silent_hold.shape))
    print("on_active_shift: ", tuple(vad_projection_head.on_active_shift.shape))
    print("on_active_hold: ", tuple(vad_projection_head.on_active_hold.shape))
    print("--------------------------------------------")
    plot_all_labels(vad_projection_head)


if __name__ == "__main__":

    from datasets_turntaking import DialogAudioDM
    from conv_ssl.ulm_projection import ULMProjection
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    import matplotlib as mpl

    mpl.use("tkagg")

    chpt = "checkpoints/wav2vec_epoch=6-val_loss=2.42136.ckpt"
    # chpt = "checkpoints/hubert_epoch=18-val_loss=1.61074.ckpt"
    model = ULMProjection.load_from_checkpoint(chpt)
    model.eval
    model.to("cuda")
    data_conf = DialogAudioDM.load_config()
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hz=model.encoder.frame_hz,
        vad_bin_times=data_conf["dataset"]["vad_bin_times"],
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=4,
        num_workers=4,
    )
    dm.prepare_data()
    dm.setup()
    diter = iter(dm.val_dataloader())

    vad_projection = VadProjection(
        bin_times=[0.2, 0.4, 0.6, 0.8],
        vad_threshold=0.5,
        pred_threshold=0.5,
        event_min_context=1.0,
        event_min_duration=0.15,
        event_horizon=1.0,
        event_start_pad=0.05,
        event_target_duration=0.10,
        # frame_hz=model.encoder.frame_hz,
        frame_hz=100,
    )
    vad_projection

    max = 100
    # metric = Metric()
    metric = ShiftHoldMetric()
    # for ii, batch in enumerate(tqdm(diter, total=max)):
    for ii, batch in enumerate(tqdm(diter)):
        # batch = next(diter)
        with torch.no_grad():
            batch = to_device(batch, model.device)
            loss, out, batch, _ = model.shared_step(batch, reduction="none")
        ret = vad_projection(out["logits_vp"], batch["vad"])
        metric.update(
            hold_correct=ret["hold"]["correct"],
            hold_total=ret["hold"]["n"],
            shift_correct=ret["shift"]["correct"],
            shift_total=ret["shift"]["n"],
        )
        # metric.update(ret)
        if ii == max:
            break
    r = metric.compute()
    print("=" * 50)
    print("F1 Weighted: ", r["f1_weighted"])
    print("-" * 35)
    print("Hold")
    for k, v in r["hold"].items():
        print(f"{k}: {v}")
    print("-" * 35)
    print("Shift")
    for k, v in r["shift"].items():
        print(f"{k}: {v}")
    print("=" * 50)

    # Probs
    plot_one = False
    if plot_one:
        probs = out["logits_vp"].softmax(dim=-1)
        vad = batch["vad"]
        p_next = vad_projection.get_next_speaker_probs(probs, vad).cpu()
        p_shift = vad_projection.speaker_prob_to_shift(p_next, vad)
        # TEST PLACES
        # Valid shift/hold
        valid = DialogEvents.find_valid_silences(
            batch["vad"],
            horizon=vad_projection.event_horizon,
            min_context=vad_projection.event_min_context,
            min_duration=vad_projection.event_min_duration,
            start_pad=vad_projection.event_start_pad,
            target_frames=vad_projection.event_target_duration,
        )
        hold, shift = DialogEvents.find_hold_shifts(batch["vad"])
        hold, shift = torch.logical_and(hold, valid.unsqueeze(-1)), torch.logical_and(
            shift, valid.unsqueeze(-1)
        )

        b = 0
        _ = plot_next_speaker_probs(
            p_next[b].cpu(),
            shift_prob=p_shift[b].cpu(),
            vad=vad[b].cpu(),
            shift=shift[b].sum(dim=-1).cpu(),
            hold=hold[b].sum(dim=-1).cpu(),
        )

    # Dataset can use 'valid' + hold/shift to only extract short
    # windows at appropriate times. may balance between shift/hold

    # Is RNN better at not be "slow" at turn-shifts?
    # what effect does sequence length have on performance?
    # How much context should be the minimum for turn-taking? 3 sec?

    # metrics
    # * one single frame
    # * aggregate
    # * aggregate + single chunk

    ################################################################################
    # b = 0
    # sil_probs = vad_projection.get_silence_shift_probs(probs).cpu()
    # act_probs = vad_projection.get_active_shift_probs(probs).cpu()
    # fig, ax = plot_vp_head(
    #     sil_probs[b], act_probs[b], vad[b], valid[b], hold[b], shift[b], plot=True
    # )

    ################################################################################
    # Hubert
    ################################################################################
    # ==================================================
    # F1 Weighted:  tensor(0.8711)
    # -----------------------------------
    # Hold
    # f1: 0.9284378886222839
    # support: 121012.0
    # precision: 0.8988163471221924
    # recall: 0.9600783586502075
    # tp: 116181.0
    # tn: 14690.0
    # fp: 13079.0
    # fn: 4831.0
    # -----------------------------------
    # Shift
    # f1: 0.6212729811668396
    # support: 27769.0
    # precision: 0.7525229454040527
    # recall: 0.5290071368217468
    # tp: 14690.0
    # tn: 116181.0
    # fp: 4831.0
    # fn: 13079.0

    ################################################################################
    # Wav2vec
    ################################################################################
    # ==================================================
    # F1 Weighted:  tensor(0.8286)
    # -----------------------------------
    # Hold
    # f1: 0.9121464490890503
    # support: 238751.0
    # precision: 0.8662189245223999
    # recall: 0.9632169008255005
    # tp: 229969.0
    # tn: 19124.0
    # fp: 35517.0
    # fn: 8782.0
    # -----------------------------------
    # Shift
    # f1: 0.4633481502532959
    # support: 54641.0
    # precision: 0.6853006482124329
    # recall: 0.34999358654022217
    # tp: 19124.0
    # tn: 229969.0
    # fp: 8782.0
    # fn: 35517.0
    # ==================================================
