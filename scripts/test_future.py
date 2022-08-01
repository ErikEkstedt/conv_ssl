from os.path import join
import torch

import matplotlib.pyplot as plt

from conv_ssl.evaluation.evaluation_phrases import load_model_dset
from conv_ssl.utils import everything_deterministic

everything_deterministic()


if __name__ == "__main__":

    ch_root = "assets/PaperB/checkpoints"
    checkpoint = join(ch_root, "cpc_48_50hz_15gqq5s5.ckpt")
    model, dset = load_model_dset(checkpoint)

    example = "student"
    d = dset.get_sample(example, "long", "female", 0)
    loss, out, probs, batch = model.output(d)

    # cutoff
    scp = [a for a in d["words"] if a[-1] == example][0]
    end_time = scp[1]
    t = d["waveform"].shape[-1] / model.sample_rate  # seconds
    n_samples = int(end_time * model.sample_rate)
    n_frames = int(end_time * model.frame_hz)
    tsil = 1
    n_sil_samples = int(tsil * model.sample_rate)
    n_sil_frames = int(tsil * model.frame_hz)
    # Voice Activity History
    vah = d["vad_history"][:, :n_frames]
    vah_sil = vah[:, -1].repeat(1, n_sil_frames, 1)
    vah = torch.cat((vah, vah_sil), dim=1)
    # Voice Activity
    va = d["vad"][:, :n_frames]  # , torch.zeros(1, n_sil_frames, 2)), dim=1)
    va_sil = torch.zeros(1, n_sil_frames, 2)
    va = torch.cat((va, va_sil), dim=1)
    va_horizon = torch.zeros(1, model.horizon_frames, 2)
    va = torch.cat((va, va_horizon), dim=1)

    d_short = {
        "waveform": torch.cat(
            (d["waveform"][:, :n_samples], torch.zeros(1, n_sil_samples)), dim=-1
        ),
        "vad": va,
        "vad_history": vah,
    }

    sloss, sout, sprobs, sbatch = model.output(d_short)

    # Compare
    n = sprobs["p"].shape[1]
    sp = sprobs["p"][0, :, 1]
    p = probs["p"][0, :n, 1]

    fig, ax = plt.subplots(4, 1)
    ax[0].plot(p, label="original", color="b")
    ax[1].plot(sp, label="cutoff", color="r")
    ax[2].plot(p, label="original", color="b")
    ax[2].plot(sp, label="cutoff", color="r")
    ax[3].plot(sp - p, label="diff")
    ax[3].set_ylim([-0.1, 0.1])
    for a in ax:
        a.legend()
        a.vlines(n_frames - 1, ymin=0, ymax=1)
    plt.show()
