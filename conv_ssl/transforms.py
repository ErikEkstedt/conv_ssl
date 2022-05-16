import torch
import torch.nn as nn

from conv_ssl.augmentations import (
    flatten_pitch_batch,
    shift_pitch_batch,
    low_pass_filter_resample,
    IntensityNeutralizer,
)


class FlatPitch(nn.Module):
    def __init__(
        self,
        target_f0: int = -1,
        statistic: str = "mean",
        stats_frame_length: int = 800,
        stats_hop_length: int = 320,
        sample_rate: int = 16000,
        to_mono: bool = True,
    ):
        super().__init__()
        self.statistic = statistic
        self.stats_frame_length = stats_frame_length
        self.stats_hop_length = stats_hop_length
        self.target_f0 = target_f0
        self.sample_rate = sample_rate
        self.to_mono = to_mono

    def forward(self, waveform, vad):
        """Appends a flipped version of the batch-samples"""
        w = flatten_pitch_batch(
            waveform=waveform,
            vad=vad,
            target_f0=self.target_f0,
            statistic=self.statistic,
            stats_frame_length=self.stats_frame_length,
            stats_hop_length=self.stats_hop_length,
            sample_rate=self.sample_rate,
            to_mono=self.to_mono,
        )
        return w


class ShiftPitch(nn.Module):
    def __init__(
        self, factor: float = 0.9, sample_rate: int = 16000, to_mono: bool = True
    ):
        super().__init__()
        self.factor = factor
        self.sample_rate = sample_rate
        self.to_mono = to_mono

    def forward(self, waveform, vad=None):
        return shift_pitch_batch(
            waveform=waveform,
            factor=self.factor,
            vad=vad,
            sample_rate=self.sample_rate,
            to_mono=self.to_mono,
        )


class LowPass(nn.Module):
    def __init__(
        self,
        cutoff_freq: int = 300,
        sample_rate: int = 16000,
        norm: bool = True,
        to_mono: bool = True,
    ):
        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.norm = norm
        self.to_mono = to_mono

    def normalize(self, x):
        assert x.ndim == 2, f"normalization expects (B, n_samples) got {x.shape}"
        xx = x - x.min(-1, keepdim=True).values
        xx = 2 * xx / xx.max()
        xx = xx - 1.0
        return xx

    def forward(self, waveform, *args, **kwargs):
        waveform = low_pass_filter_resample(
            waveform, self.cutoff_freq, self.sample_rate
        )
        if self.to_mono:
            waveform = waveform.mean(1)

        if self.norm:
            waveform = self.normalize(waveform)

        return waveform


class FlatIntensity(nn.Module):
    """ """

    def __init__(
        self,
        vad_hz,
        vad_cutoff: float = 0.2,
        hop_time: float = 0.01,
        f0_min: int = 60,
        statistic: str = "mean",
        sample_rate: int = 16000,
        to_mono: bool = True,
    ):
        super().__init__()
        self.hop_time = hop_time
        self.vad_hz = vad_hz
        self.f0_min = f0_min
        self.vad_cutoff = vad_cutoff
        self.statistic = statistic
        self.sample_rate = sample_rate
        self.to_mono = to_mono
        self.neutralizer = IntensityNeutralizer(
            hop_time=hop_time,
            vad_hz=vad_hz,
            f0_min=f0_min,
            vad_cutoff=vad_cutoff,
            scale_stat=statistic,
            sample_rate=sample_rate,
            to_mono=to_mono,
        )

    def forward(self, waveform, vad):
        combine = False
        if waveform.ndim == 3:
            combine = True
        if combine:
            y_tmp = waveform.mean(1)
        else:
            y_tmp = waveform
        y, _ = self.neutralizer(y_tmp, vad=vad)
        return y
