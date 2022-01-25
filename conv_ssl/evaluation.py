from os import cpu_count

import torch
import pytorch_lightning as pl

from conv_ssl.ulm_projection import ULMProjection
from conv_ssl.models import ProjectionMetricCallback, ProjectionMetrics
from datasets_turntaking import DialogAudioDM

import matplotlib.pyplot as plt
from datasets_turntaking.features.plot_utils import plot_melspectrogram, plot_vad_oh

import math
import torch.nn.functional as F

from datasets_turntaking.features.vad import VAD


def gaussian_kernel_1d_unidirectional(N, sigma=3):
    """

    N: points to include in smoothing (including current) i.e. N=5 => [t-4, t-3, t-2, t-1, 5]

    source:
        https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351
    """
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    kernel_size = (N - 1) * 2 + 1
    x_cord = torch.arange(kernel_size).unsqueeze(0)
    mean = (kernel_size - 1) / 2.0
    variance = sigma ** 2.0

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    scale = 1.0 / (2.0 * math.pi * variance)
    gaussian_kernel = scale * torch.exp(
        -torch.sum((x_cord - mean) ** 2.0, dim=0) / (2 * variance)
    )

    # only care about left half
    gaussian_kernel = gaussian_kernel[:N]

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return gaussian_kernel


def smooth_gaussian(x, N, sigma=1):
    kernel = gaussian_kernel_1d_unidirectional(N, sigma)

    # pad left
    xx = F.pad(x, pad=(N - 1, 0), mode="replicate")
    if xx.ndim == 2:
        xx = xx.unsqueeze(1)
    a = F.conv1d(xx, weight=kernel.view(1, 1, -1), bias=None, stride=1)
    return a.squeeze(1)


if __name__ == "__main__":

    # from conv_ssl.eval import manual_eval
    import matplotlib as mpl

    mpl.use("TkAgg")

    chpt = "checkpoints/wav2vec_epoch=6-val_loss=2.42136.ckpt"
    # chpt = "checkpoints/hubert_epoch=18-val_loss=1.61074.ckpt"
    model = ULMProjection.load_from_checkpoint(chpt)
    model.eval

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
        num_workers=cpu_count(),
    )
    dm.prepare_data()
    dm.setup()
    # trainer = pl.Trainer(limit_val_batches=100, callbacks=[ProjectionMetricCallback()], gpus=-1)
    # # result = trainer.test(model, dataloaders=dm.val_dataloader())
    # result = trainer.validate(model, dataloaders=dm.val_dataloader())
    # result = manual_eval(model.to("cuda"), dm.val_dataloader(), max_iter=100)
    # for k, v in result["result"].items():
    #     print(f"{k}: {v}")
    # samples = result["samples"]
    # low_high = "low"
    # model.animate_sample(
    #     waveform=samples[low_high][0]["waveform"][0].to(model.device),
    #     vad=samples[low_high][0]["vad"][0].to(model.device),
    #     path="ulm_projection_vid.mp4",
    # )
    diter = iter(dm.val_dataloader())

    # smooth_time = 1.0
    # smooth_time_sigma = 0.1
    # vad_hop_time = 1.0 / model.encoder.frame_hz
    # smooth_frames = int(smooth_time / vad_hop_time)
    # smooth_sigma = int(smooth_frames * (smooth_time_sigma / smooth_time))
    # print("smooth_frames: ", smooth_frames)
    # print("smooth_sigma: ", smooth_sigma)

    def get_shift_probs(out, batch, model):
        next_speaker = model.vad_projection_codebook.get_next_speaker(
            batch["vad_label"]
        )
        next_speaker_probs = model.vad_projection_codebook.first_speaker_probs(
            logits=out["logits_vp"]
        )
        hold_oh, shift_oh = VAD.get_hold_shift_onehot(batch["vad"])

        holds = torch.where(hold_oh)
        next_speaker_hold = next_speaker[holds]
        hold_probs = next_speaker_probs[holds[0], holds[1], next_speaker_hold]

        shifts = torch.where(shift_oh)
        next_speaker_shift = next_speaker[shifts]
        shift_probs = next_speaker_probs[shifts[0], shifts[1], next_speaker_shift]

        turn_shift_probs = torch.zeros_like(next_speaker, dtype=torch.float)
        turn_shift_probs[holds] = 1.0 - hold_probs
        turn_shift_probs[shifts] = shift_probs
        return turn_shift_probs

    batch = next(diter)
    with torch.no_grad():
        loss, out, batch, _ = model.shared_step(batch, reduction="none")
        ent = torch.distributions.categorical.Categorical(
            logits=out["logits_vp"]
        ).entropy()
        ent /= 5.5452
        probs = model.vad_projection_codebook.get_hold_shift_probs(
            logits=out["logits_vp"], vad=batch["vad"]
        )
        # ent_smooth = smooth_gaussian(ent, smooth_frames, sigma=smooth_sigma)
    # Shift/Hold probs
    # speaker_probs = self.first_speaker_probs(logits, probs)
    # last_speaker = VAD.get_last_speaker(vad)

    b = 3
    plt.close("all")
    f, ax = plot_static(batch["waveform"][b], vad=batch["vad"][b], plot=False)
    # plot_entropy(ent[b], ax[-1])
    plot_entropy(loss["vp"][b], ax[-1], label="celoss", color="k", ylim=None)
    plot_probs(probs["shift"][b], ax[-2])
    plt.show()

    # diter = iter(dm.val_dataloader())
    # batch = next(diter)
    # model.animate_sample(
    #     waveform=batch["waveform"][1],
    #     vad=batch["vad"][1],
    #     path="ulm_projection_vid.mp4",
    # )
