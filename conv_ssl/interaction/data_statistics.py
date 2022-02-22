import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets_turntaking import DialogAudioDM
from conv_ssl.utils import find_island_idx_len
from vad_turn_taking import VadLabel, ProjectionCodebook, DialogEvents as DE


def extract_statistics(
    dloader,
    min_duration=5,
    horizon=100,
    bc_pre_silence=150,
    bc_post_silence=300,
    bc_max_active=100,
    VL=None,
    PC=None,
    total=None,
):
    stats = {
        "hold": {"n": 0, "duration": []},
        "shift": {"n": 0, "duration": []},
        "bc": {"n": 0, "duration": []},
    }

    if PC is not None:
        stats["discrete"] = torch.zeros((PC.n_classes))

    if total is None:
        total = len(dloader)

    for batch_idx, batch in enumerate(tqdm(dloader, total=total)):
        if batch_idx == total:
            break
        batch_size = batch["vad"].shape[0]
        hold, shift = DE.on_silence(
            batch["vad"],
            start_pad=0,
            target_frames=-1,
            horizon=horizon,
            min_context=0,
            min_duration=min_duration,
        )
        backchannels = DE.extract_bc_candidates(
            batch["vad"],
            pre_silence_frames=bc_pre_silence,
            post_silence_frames=bc_post_silence,
            max_active_frames=bc_max_active,
        )

        if VL is not None:
            vad_projection_window = VL.vad_projection(batch["vad"])
            discrete_labels = PC(vad_projection_window)
            # -> onehot
            discrete_labels = F.one_hot(discrete_labels, num_classes=PC.n_classes).view(
                -1, PC.n_classes
            )
            discrete_labels = discrete_labels.sum(dim=0)  # sum over all labels
            stats["discrete"] += discrete_labels

        for b in range(batch_size):
            for speaker in [0, 1]:
                start, dur, val = find_island_idx_len(hold[b, :, speaker])
                start = start[val]
                if len(start) > 0:
                    dur = dur[val]
                    stats["hold"]["n"] += len(dur)
                    stats["hold"]["duration"] += dur.tolist()

                start, dur, val = find_island_idx_len(shift[b, :, speaker])
                start = start[val]
                if len(start) > 0:
                    dur = dur[val]
                    stats["shift"]["n"] += len(dur)
                    stats["shift"]["duration"] += dur.tolist()

                start, dur, val = find_island_idx_len(backchannels[b, :, speaker])
                start = start[val == 1]
                if len(start) > 0:
                    dur = dur[val == 1]
                    stats["bc"]["n"] += len(dur)
                    stats["bc"]["duration"] += dur.tolist()
    return stats


def plot_hold_shift_histogram(stats, plot=False):
    fig, ax = plt.subplots(1, 1)
    nh, bins, line = ax.hist(
        stats["hold"]["duration"],
        color="r",
        label=f'Hold ({stats["hold"]["n"]})',
        alpha=0.6,
        bins=100,
    )
    b = ax.hist(
        stats["shift"]["duration"],
        color="g",
        label=f'Shift ({stats["shift"]["n"]})',
        alpha=1.0,
        bins=bins,
    )
    c = ax.hist(
        stats["bc"]["duration"],
        color="k",
        label=f'bc ({stats["bc"]["n"]})',
        alpha=1.0,
        bins=bins,
    )
    ax.legend(loc="upper right")
    if plot:
        plt.pause(0.1)
    return fig, ax


if __name__ == "__main__":

    frame_hz = 100
    bin_times = [0.2, 0.4, 0.6, 0.8]
    batch_size = 16
    num_workers = 4

    data_conf = DialogAudioDM.load_config()
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hz=frame_hz,
        vad_horizon=sum(bin_times),
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        flip_channels=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    dm.prepare_data()
    dm.setup()

    VL = VadLabel(bin_times=bin_times, vad_hz=frame_hz, threshold_ratio=0.5)
    PC = ProjectionCodebook(bin_times=bin_times, frame_hz=frame_hz)
    stats = extract_statistics(dm.val_dataloader(), VL=VL, PC=PC)
    torch.save(stats, "assets/val_stats.pt")

    q_prior = stats["discrete"] / stats["discrete"].sum()
    p_log_p = q_prior * q_prior.log2()
    ent = -p_log_p.sum(-1)
    print("entropy: ", ent)

    # silence
    A_next = PC.on_silent_shift[0]
    B_next = PC.on_silent_shift[1]
    AN = q_prior[A_next].sum()
    BN = q_prior[B_next].sum()
    AB_other = 1 - AN - BN
    qq = torch.tensor([AN, BN, AB_other])
    p_log_p = qq * qq.log()
    ent = -p_log_p.sum(dim=-1)
    print("qq: ", qq)
    print("ent: ", ent)
    # active
    A_anext = PC.on_active_shift[0]
    B_anext = PC.on_active_shift[1]
    AAN = q_prior[A_anext].sum()
    BBN = q_prior[B_anext].sum()
    AABB_other = 1 - AAN - BBN
    qq2 = torch.tensor([AAN, BBN, AABB_other])
    p_log_p2 = qq2 * qq2.log()
    ent2 = -p_log_p2.sum(dim=-1)
    print("qq2: ", qq2)
    print("ent2: ", ent2)

    N = 256
    d, idx = q_prior.sort(descending=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.bar(
        torch.arange(len(d[:N])),
        d[:N],
        label=f"Entropy: {round(ent.item(),2)} bits",
    )
    ax.bar(A_next, d[A_next], color="red", label="A is next")
    ax.bar(B_next, d[B_next], color="orange", label="B is next")
    ax.legend(loc="upper right")
    ax.set_ylabel("log probabilitiy")
    ax.set_xlabel("Projection window, Rank")
    ax.hlines(y=1 / 256, xmin=0, xmax=N)
    ax.set_yscale("log")
    fig.suptitle("Validation Set")
    # ax.set_xticks(list(range(len(d[:N]))))
    # ax.set_xticklabels(idx[:N].tolist(), fontsize=8, rotation=45)
    plt.tight_layout()
    plt.show()

    from torch.distributions import Categorical

    ent = Categorical(probs=q_prior).entropy()

    uni_ent = Categorical(probs=torch.ones(256) / 256).entropy()
    ent = qd.entropy()

    stats = torch.load("assets/val_stats.pt")

    # stats = extract_statistics(dm.val_dataloader())
    # torch.save(stats, "assets/val_stats.pt")
    # stats = extract_statistics(dm.train_dataloader())
    # torch.save(stats, "assets/train_stats.pt")
    # tot = stats["hold"]["n"] + stats["shift"]["n"] + stats["bc"]["n"]
    # rh = stats["hold"]["n"] / tot
    # rs = stats["shift"]["n"] / tot
    # rb = stats["bc"]["n"] / tot
    # rbs = (stats["bc"]["n"] + stats["shift"]["n"]) / tot
    # print("Holds: ", stats["hold"]["n"])
    # print("Shift: ", stats["shift"]["n"])
    # print("BC: ", stats["bc"]["n"])
    # print("Hold %: ", round(rh, 3))
    # print("Shift %: ", round(rs, 3))
    # print("BC %: ", round(rb, 3))
    # print("BC+Shift %: ", round(rbs, 3))
    # fig, ax = plot_hold_shift_histogram(stats)
    # plt.show()

    # Different training metrics
    VL = VadLabel(bin_times=bin_times, vad_hz=frame_hz, threshold_ratio=0.5)
    PC = ProjectionCodebook(bin_times=bin_times, frame_hz=frame_hz)

    batch = next(iter(dm.val_dataloader()))

    v_compare = VL.comparative_activity(batch["vad"])
    v_independent = VL.vad_projection(batch["vad"])
    v_discrete = PC.onehot_to_idx(v_independent)

    fig, ax = plt.subplots(1, 1)
    ax.hist(c.unsqueeze(0), range=(0, 1), bins=20)
    plt.show()
