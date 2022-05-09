import torch
import torchaudio.functional as AF
import einops
import numpy as np
from typing import Union, Optional

import parselmouth
from parselmouth.praat import call

from datasets_turntaking.utils import load_waveform


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
        if isinstance(waveform_or_path, torch.Tensor):
            waveform_or_path = waveform_or_path.numpy()

        if waveform_or_path.dtype != np.float64:
            waveform_or_path = waveform_or_path.astype("float64")
        sound = parselmouth.Sound(waveform_or_path, sampling_frequency=sample_rate)

    manipulation = call(sound, "To Manipulation", hop_time, f0_min, f0_max)

    if isinstance(target_f0, str):
        pitch_tier = call(manipulation, "Extract pitch tier")
        if target_f0 == "mean":
            pass
        elif target_f0 == "median":
            pass

        target_f0 = 200
        raise NotADirectoryError("median/mean pitch not done")

    # Flat pitch
    pitch_tier = create_flat_pitch_tier(
        manipulation, target_f0, sound.start_time, sound.end_time
    )

    # Select the original and the replacement tier -> replace pitch
    call([pitch_tier, manipulation], "Replace pitch tier")

    # Extract the new sound
    sound_flat = call(manipulation, "Get resynthesis (overlap-add)")

    # To Array
    y_flat = sound_flat.as_array().astype("float32")
    return torch.from_numpy(y_flat)


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

    # Calculate F0 averages for each channel/speaker
    w = einops.rearrange(waveform, "b c n -> (b c) n")
    f0 = f0_kaldi_torch(
        w, sr=sample_rate, frame_length=frame_length, hop_length=hop_length
    )
    f0 = einops.rearrange(f0, "(b c) n -> b c n", c=2)

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

    # From filename (Slower)
    # t = time.time()
    # y_flat = flatten_pitch_praat(
    #     SAMPLE, target_f0=150, hop_time=hop_time, sample_rate=sample_rate
    # )
    # t = round(time.time() - t, 3)
    # print(f"All Praat: {t}")


if __name__ == "__main__":
    _parselmouth_example()
