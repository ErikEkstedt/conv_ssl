import torch
import torchaudio.functional as AF
import torchaudio.transforms as AT
import einops
import numpy as np
from typing import Union, Optional

import parselmouth
from parselmouth.praat import call

from datasets_turntaking.utils import load_waveform
from conv_ssl.models.cnn import CConv1d


"""
* Praat: https://www.fon.hum.uva.nl/praat/
* Flatten pitch script: http://phonetics.linguistics.ucla.edu/facilities/acoustic/FlatIntonationSynthesizer.txt
* Parselmouth: https://parselmouth.readthedocs.io/en/latest/examples/pitch_manipulation.html
"""


def f0_kaldi_torch(
    y: torch.Tensor,
    sr: int = 16000,
    fmin: int = 60,
    fmax: int = 400,
    frame_length: int = 400,
    hop_length: int = 200,
    **kwargs,
) -> torch.Tensor:
    frame_length_ms = 1000 * frame_length / sr
    hop_length_ms = 1000 * hop_length / sr
    f0 = AF.compute_kaldi_pitch(
        y,
        sample_rate=sr,
        frame_length=frame_length_ms,
        frame_shift=hop_length_ms,
        min_f0=fmin,
        max_f0=fmax,
        **kwargs,
    )
    return f0[..., 1]


def torch_to_praat_sound(x, sample_rate):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    if x.dtype != np.float64:
        x = x.astype("float64")
    return parselmouth.Sound(x, sampling_frequency=sample_rate)


def praat_to_torch(sound):
    y = sound.as_array().astype("float32")
    return torch.from_numpy(y)


###############################################
###################### F0 #####################
###############################################
def create_flat_pitch_tier(manipulation, target_f0: float, start: float, end: float):
    """Flat pitch tier"""
    pitch_tier = call(manipulation, "Create PitchTier", "flat", start, end)
    call(pitch_tier, "Add point", start, target_f0)
    call(pitch_tier, "Add point", end, target_f0)
    return pitch_tier


def flatten_pitch_praat(
    waveform_or_path: Union[str, torch.Tensor, np.ndarray],
    target_f0: Union[float, str] = 200,
    hop_time: float = 0.01,
    f0_min: int = 60,
    f0_max: int = 400,
    sample_rate: int = 16000,
):
    if isinstance(waveform_or_path, str):
        sound = parselmouth.Sound(waveform_or_path)
    else:
        sound = torch_to_praat_sound(waveform_or_path, sample_rate)

    manipulation = call(sound, "To Manipulation", hop_time, f0_min, f0_max)

    if isinstance(target_f0, str):
        pitch_tier = call(manipulation, "Extract pitch tier")
        if target_f0 == "mean":
            pass
        elif target_f0 == "median":
            pass

        target_f0 = 200
        raise NotImplementedError("median/mean pitch not done")

    # Flat pitch
    pitch_tier = create_flat_pitch_tier(
        manipulation, target_f0, sound.start_time, sound.end_time
    )

    # Select the original and the replacement tier -> replace pitch
    call([pitch_tier, manipulation], "Replace pitch tier")

    # Extract the new sound
    sound_flat = call(manipulation, "Get resynthesis (overlap-add)")
    return praat_to_torch(sound_flat)


def shift_pitch_praat(
    waveform_or_path: Union[str, torch.Tensor, np.ndarray],
    factor: float = 1.1,
    hop_time: float = 0.01,
    f0_min: int = 60,
    f0_max: int = 400,
    sample_rate: int = 16000,
):
    if isinstance(waveform_or_path, str):
        sound = parselmouth.Sound(waveform_or_path)
    else:
        sound = torch_to_praat_sound(waveform_or_path, sample_rate)

    # Source: https://parselmouth.readthedocs.io/en/latest/examples/pitch_manipulation.html
    manipulation = call(sound, "To Manipulation", hop_time, f0_min, f0_max)
    pitch_tier = call(manipulation, "Extract pitch tier")

    call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor)
    call([pitch_tier, manipulation], "Replace pitch tier")

    sound_shifted = call(manipulation, "Get resynthesis (overlap-add)")
    return praat_to_torch(sound_shifted)


def get_f0_statistic(
    waveform,
    statistic="mean",
    frame_length: int = 800,
    hop_length: int = 320,
    sample_rate: int = 16000,
):
    valid_stats = ["mean", "median"]
    assert waveform.ndim == 3, f"Expects (B, C, n_samples) got {tuple(waveform.shape)}"
    assert (
        statistic in valid_stats
    ), f"`statistic` must be one of {valid_stats} got {statistic}"

    nb, c, _ = waveform.shape

    # Calculate F0 averages for each channel/speaker
    w = einops.rearrange(waveform, "b c n -> (b c) n")
    f0 = f0_kaldi_torch(
        w, sr=sample_rate, frame_length=frame_length, hop_length=hop_length
    )
    f0 = einops.rearrange(f0, "(b c) n -> b c n", c=c)

    if statistic == "mean":
        target_f0 = f0.mean(-1).round()
    else:
        target_f0 = f0.median(-1).values.round()

    return target_f0


def flatten_pitch_batch(
    waveform,
    vad: Optional[torch.Tensor] = None,
    target_f0: int = -1,
    statistic: str = "mean",
    stats_frame_length: int = 800,
    stats_hop_length: int = 320,
    sample_rate: int = 16000,
    to_mono: bool = True,
) -> torch.Tensor:
    valid_stats = ["mean", "median"]
    assert waveform.ndim == 3, f"Expects (B, C, n_samples) got {tuple(waveform.shape)}"
    assert (
        statistic in valid_stats
    ), f"`statistic` must be one of {valid_stats} got {statistic}"

    batch_size = waveform.shape[0]
    channels = waveform.shape[1]

    # extract target f0
    if target_f0 > 0:
        target_f0s = torch.tensor((batch_size, channels)).fill_(target_f0)
    else:
        target_f0s = get_f0_statistic(
            waveform,
            statistic,
            frame_length=stats_frame_length,
            hop_length=stats_hop_length,
            sample_rate=sample_rate,
        )

    # Don't change pitch where there is no activity
    if vad is not None:
        active = vad.sum(1).cpu()
    else:
        active = torch.ones((batch_size, channels))

    flat = []
    for nb in range(batch_size):
        flat_batch = []
        for ch in range(channels):
            y = waveform[nb, ch].unsqueeze(0)  # add batch/channel dim
            if active[nb, ch] == 0:
                flat_batch.append(y)
            else:
                tgt_f0 = target_f0s[nb, ch].item()
                y_tmp = flatten_pitch_praat(y, tgt_f0, sample_rate=sample_rate)
                flat_batch.append(y_tmp)
        flat_batch = torch.cat(flat_batch)
        flat.append(flat_batch)
    flat = torch.stack(flat)

    # to mono
    if to_mono:
        flat = flat.mean(1)
    return flat


def shift_pitch_batch(
    waveform,
    factor: float,
    vad: Optional[torch.Tensor] = None,
    sample_rate: int = 16000,
    to_mono: bool = True,
) -> torch.Tensor:
    assert waveform.ndim == 3, f"Expects (B, C, n_samples) got {tuple(waveform.shape)}"

    batch_size = waveform.shape[0]
    channels = waveform.shape[1]

    # Don't change pitch where there is no activity
    if vad is not None:
        active = vad.sum(1).cpu()
    else:
        active = torch.ones((batch_size, channels))

    flat = []
    for nb in range(batch_size):
        flat_batch = []
        for ch in range(channels):
            y = waveform[nb, ch].unsqueeze(0)  # add batch/channel dim
            if active[nb, ch] == 0:
                flat_batch.append(y)
            else:
                y_tmp = shift_pitch_praat(y, factor=factor, sample_rate=sample_rate)
                flat_batch.append(y_tmp)
        flat_batch = torch.cat(flat_batch)
        flat.append(flat_batch)
    flat = torch.stack(flat)

    # to mono
    if to_mono:
        flat = flat.mean(1)
    return flat


def low_pass_filter_resample(x, cutoff_freq, sample_rate=16000):
    new_freq = int(cutoff_freq * 2)  # Nyquist
    x = AF.resample(x, orig_freq=sample_rate, new_freq=new_freq)
    x = AF.resample(x, orig_freq=new_freq, new_freq=sample_rate)
    return x


@torch.no_grad()
def smooth(x, kernel_size=15):
    averager = CConv1d(
        in_channels=1, out_channels=1, kernel_size=kernel_size, bias=False
    )
    averager.weight.data.fill_(1 / kernel_size)
    averager.eval()
    if x.ndim == 2:
        x = x.unsqueeze(1)
    return averager(x)[:, 0]


def intensity_mask(intensity, cutoff=0.2, kernel_size=15):
    ints = smooth(intensity, kernel_size)
    ints[ints < 0] = 0
    ii = ints - ints.min()
    ii = ii / ii.max()
    return ii < cutoff


def praat_intensity(y, f0_min=60, hop_time=0.01, sample_rate=16000):
    sound = torch_to_praat_sound(y, sample_rate=sample_rate)
    intensity = sound.to_intensity(
        minimum_pitch=f0_min, time_step=hop_time, subtract_mean=False
    )
    return praat_to_torch(intensity)


class IntensityNeutralizer(object):
    def __init__(
        self,
        hop_time=0.01,
        vad_hz=None,
        f0_min=60,
        vad_cutoff=0.2,
        scale_stat="mean",
        sample_rate: int = 16000,
        kernel_size: int = 15,
        to_mono: bool = True,
    ):
        self.hop_time = hop_time
        self.f0_min = f0_min
        self.vad_cutoff = vad_cutoff
        self.vad_hz = vad_hz
        self.scale_stat = scale_stat
        self.sample_rate = sample_rate
        self.kernel_size = kernel_size
        self.smoother = self._smoother(kernel_size)
        self.to_mono = to_mono

    def _smoother(self, kernel_size):
        smoother = CConv1d(
            in_channels=1, out_channels=1, kernel_size=kernel_size, bias=False
        )
        smoother.weight.data.fill_(1 / kernel_size)
        for p in smoother.parameters():
            p.requires_grad_(False)
        return smoother

    @torch.no_grad()
    def smooth(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 1:
            x = x.unsqueeze(1).unsqueeze(1)
        return self.smoother(x)[0, 0]

    def intensity_vad(self, intensity):
        # intensity (1, n_frames)
        ints = self.smooth(intensity)
        ints[ints < 0] = 0
        ii = ints - ints.min()
        ii = ii / ii.max()
        return ii >= self.vad_cutoff

    def neutralize_intensity(self, waveform):
        assert (
            waveform.ndim == 1
        ), f"expects waveform (n_samples, ) got {waveform.shape}"

        sound = torch_to_praat_sound(waveform, sample_rate=self.sample_rate)
        intensity = sound.to_intensity(
            minimum_pitch=self.f0_min, time_step=self.hop_time, subtract_mean=False
        )
        ints = praat_to_torch(intensity)  # .as_array()
        vad = self.intensity_vad(ints).unsqueeze(0)

        mi = ints[ints > 0]
        if len(mi) < 1:
            return waveform, None

        if self.scale_stat == "max":
            m = mi.max().value.item()
        else:
            # m = intensity.get_average()
            m = mi.mean().item()

        t = [
            intensity.get_time_from_frame_number(f + 1)
            for f in range(intensity.get_number_of_frames())
        ]
        int_tier = call(
            intensity, "Create IntensityTier", "intensity_tier", 0, sound.total_duration
        )
        last_t = 0
        last_valid = False
        for t, i, valid in zip(t, ints[0], vad[0]):
            if not valid:
                if last_valid:
                    call(int_tier, "Add point", t, 0)
                last_valid = False
                continue
            if not last_valid:
                call(int_tier, "Add point", t - 0.001, 0)
            scale = m - i.item()
            call(int_tier, "Add point", t, scale)
            last_t = t
            last_valid = True
        call(int_tier, "Add point", last_t + 0.001, 0)
        new_sound = call([int_tier, sound], "Multiply", 1)
        call(new_sound, "Scale intensity", m)
        return praat_to_torch(new_sound), vad

    def __call__(self, waveform: torch.Tensor, vad=None):
        v1 = 1.0
        v0 = 1.0
        if vad.ndim == 3:
            vad = vad.squeeze(0)

        if vad is not None and self.vad_hz is not None:
            v0 = AF.resample(
                vad[:, 0], orig_freq=self.vad_hz, new_freq=self.sample_rate
            )
            v1 = AF.resample(
                vad[:, 1], orig_freq=self.vad_hz, new_freq=self.sample_rate
            )

        min_frames = min(len(v0), waveform.shape[-1])

        if waveform.shape[0] > 1:
            if isinstance(v0, torch.Tensor) and v0.sum() == 0:
                y0, m0 = waveform[0], v0
            else:
                if isinstance(v0, torch.Tensor):
                    waveform[0, :min_frames] *= v0[:min_frames]
                y0, m0 = self.neutralize_intensity(waveform[0])
                # y0, m0 = self.neutralize_intensity(waveform[0] * v0[:min_frames])

            if isinstance(v1, torch.Tensor) and v1.sum() == 0:
                y1, m1 = waveform[1], v1
            else:
                if isinstance(v1, torch.Tensor):
                    waveform[1, :min_frames] *= v1[:min_frames]
                y1, m1 = self.neutralize_intensity(waveform[1])
                # y1, m1 = self.neutralize_intensity(waveform[1, :min_frames] * v1[:min_frames])

            # print("y0: ", tuple(y0.shape))
            # print("y1: ", tuple(y1.shape))
            yy = torch.stack((y0, y1), dim=1)
            m = torch.stack((m0, m1), dim=1)
            if self.to_mono:
                yy = yy.mean(1, keepdim=True)
        else:
            v = torch.logical_or(v0, v1).float()
            if v.sum() == 0:
                yy, m = waveform, None
            else:
                waveform[0, :min_frames] *= v[:min_frames]
                yy, m = self.neutralize_intensity(waveform[0])
            # print("yy: ", tuple(yy.shape))

        yy = yy.to(waveform.device)

        if not isinstance(v1, float):
            w = torch.where(yy == 0)
            yy[w] = yy[w] + waveform[w]
        return yy, m


#################################################################################
def _parselmouth_example():
    # https://parselmouth.readthedocs.io/en/latest/examples/pitch_manipulation.html
    import time
    import sounddevice as sd
    import matplotlib.pyplot as plt

    def f0_kaldi_torch(
        y, sr, fmin=60, fmax=400, frame_length=400, hop_length=200, **kwargs
    ):
        frame_length_ms = 1000 * frame_length / sr
        hop_length_ms = 1000 * hop_length / sr
        f0 = AF.compute_kaldi_pitch(
            y,
            sample_rate=sr,
            frame_length=frame_length_ms,
            frame_shift=hop_length_ms,
            min_f0=fmin,
            max_f0=fmax,
            **kwargs,
        )
        return f0[..., 1], f0[..., 0]

    SAMPLE = "assets/audio/her.wav"

    # From torch/numpy array (Faster, especially with downsample)
    # Load waveform
    sample_rate = 16000
    hop_time = 0.01
    t = time.time()
    y, _ = load_waveform(
        path=SAMPLE,
        sample_rate=sample_rate,
        start_time=8,
        end_time=10,
        normalize=True,
        mono=True,
        audio_normalize_threshold=0.05,
    )

    # Flatten
    y_flat = flatten_pitch_praat(
        y, target_f0=209, hop_time=hop_time, sample_rate=sample_rate
    )
    t = round(time.time() - t, 3)
    print(f"Torch + praat : {t}")

    orig, _ = f0_kaldi_torch(y, sr=sample_rate)
    flat, _ = f0_kaldi_torch(y_flat, sr=sample_rate)

    sd.play(y_flat[0], samplerate=sample_rate)
    fig, ax = plt.subplots(1, 1)
    ax.plot(orig[0], color="g", label="orig")
    ax.plot(flat[0], color="r", label="flat")
    ax.legend()
    plt.show()


def _filter_test():
    from conv_ssl.evaluation.phrase_dataset import PhraseDataset
    from conv_ssl.model import VPModel
    from conv_ssl.plot_utils import plot_melspectrogram
    import matplotlib.pyplot as plt

    # checkpoint = "assets/PaperB/checkpoints/cpc_48_50hz_15gqq5s5.ckpt"
    # checkpoint = "assets/PaperB/checkpoints/cpc_48_20hz_2ucueis8.ckpt"
    checkpoint = "assets/PaperB/checkpoints/cpc_44_100hz_unfreeze_12ep.ckpt"
    model = VPModel.load_from_checkpoint(checkpoint)
    model = model.eval()
    _ = model.to("cuda")

    phrase_path = "assets/phrases_beta/phrases.json"
    dset = PhraseDataset(
        phrase_path,
        vad_hz=model.frame_hz,
        sample_rate=model.sample_rate,
        vad_horizon=model.VAP.horizon,
        vad_history=model.conf["data"]["vad_history"],
        vad_history_times=model.conf["data"]["vad_history_times"],
    )

    # sample = dset.get_sample("student", "short", "female", 0)
    sample = dset.get_sample("student", "long", "female", 0)

    # wf = AF.lowpass_biquad(sample['waveform'], sample_rate=sample_rate, cutoff_freq=250, Q=10)
    wf = AF.resample(sample["waveform"], orig_freq=16000, new_freq=500)
    wf = AF.resample(wf, orig_freq=500, new_freq=16000)

    wf = low_pass_filter_resample(
        sample["waveform"], cutoff_freq=300, sample_rate=16000
    )
    fig, ax = plt.subplots(2, 1)
    plot_melspectrogram(sample["waveform"][0], ax=ax[0])
    plot_melspectrogram(wf[0], ax=ax[1])
    plt.pause(0.1)
    sd.play(wf[0], samplerate=sample_rate)


def _intensity_test():

    import matplotlib.pyplot as plt
    import sounddevice as sd

    SAMPLE = "assets/audio/her.wav"
    tts_sample = "assets/phrases_beta/audio/basketball_long_female_en-US-Wavenet-C.wav"

    # From torch/numpy array (Faster, especially with downsample)
    # Load waveform
    sample_rate = 16000
    hop_time = 0.01
    y, _ = load_waveform(
        path=SAMPLE,
        sample_rate=sample_rate,
        start_time=8,
        end_time=10,
        normalize=False,
        mono=True,
        audio_normalize_threshold=0.05,
    )

    neutral = IntensityNeutralizer(hop_time=0.01, f0_min=60, sample_rate=16000)

    batch = dset.get_sample("student", "long", "female", 4)
    y = batch["waveform"]
    y1, mask = neutral(y)

    i = praat_intensity(y)
    i1 = praat_intensity(y)
    print("i: ", tuple(i.shape))
    print("i1: ", tuple(i1.shape))
    fig, ax = plt.subplots(1, 1)
    ax.plot(y[0], label="orig", color="b", alpha=0.5)
    ax.legend()
    ax.plot(y1[0], label="new", color="red")
    plt.pause(0.1)

    sd.play(y[0], samplerate=16000)
    sd.play(y1[0], samplerate=16000)


if __name__ == "__main__":
    _parselmouth_example()
