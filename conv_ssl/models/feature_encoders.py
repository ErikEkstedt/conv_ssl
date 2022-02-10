import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as AT

from einops.layers.torch import Rearrange
from conv_ssl.models.causal_conv import CausalConv1d
from conv_ssl.utils import repo_root, load_config


torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True)


def compute_deltas_causal(specgram: torch.Tensor, win_length: int = 5) -> torch.Tensor:
    device = specgram.device
    dtype = specgram.dtype

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape(1, -1, shape[-1])

    assert win_length >= 3

    n = (win_length - 1) // 2

    # twice sum of integer squared
    denom = n * (n + 1) * (2 * n + 1) / 3
    pad = win_length - 1
    specgram = F.pad(specgram, (pad, 0), mode="replicate")
    kernel = torch.arange(-n, n + 1, 1, device=device, dtype=dtype).repeat(
        specgram.shape[1], 1, 1
    )
    output = F.conv1d(specgram, kernel, groups=specgram.shape[1]) / denom

    # unpack batch
    output = output.reshape(shape)
    return output

    def forward(self, input):
        return super(CausalConv1d, self).forward()


class MFCCEncoder(nn.Module):
    def __init__(
        self, n_mfcc=12, delta=2, hop_time=0.01, sample_rate=16000, n_mels=12
    ) -> None:
        super().__init__()

        self.hop_time = hop_time
        self.sample_rate = sample_rate
        self.n_fft = int(sample_rate * hop_time)
        self.hop_length = int(sample_rate * hop_time)

        self.n_mfcc = n_mfcc
        self.delta = delta

        self.dim = self.n_mfcc + int(self.n_mfcc * delta)

        melkwargs = {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": n_mels,
        }
        self.mfcc = AT.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs=melkwargs)

    def forward(self, waveform):
        mfcc = self.mfcc(waveform)
        if self.delta > 0:
            mfcc = [mfcc]
            for _ in range(self.delta):
                mfcc.append(compute_deltas_causal(mfcc[-1]))
            mfcc = torch.cat(mfcc, dim=1)
        return mfcc


class FeatureEncoder(nn.Module):
    @staticmethod
    def load_config(path=None, args=None, format="dict"):
        if path is None:
            path = repo_root() + "/conv_ssl/config/model.yaml"
        return load_config(path, args=args, format=format)

    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf

        in_dim = 1
        if conf["encoder"]["features"] == "mfcc":
            self.feature_encoder = nn.Sequential(
                MFCCEncoder(), Rearrange("... c t -> ... t c")
            )
            in_dim = self.feature_encoder[0].dim
        elif conf["encoder"]["features"] == "melspec":
            self.feature_encoder = MelSpecEncoder()
        else:
            self.feature_encoder = nn.Identity()

        if conf["encoder"]["type"] == "lstm":
            self.processor = nn.LSTM(in_dim, 64, batch_first=True)
        elif conf["encoder"]["type"] == "gru":
            self.processor = nn.GRU(in_dim, 64, batch_first=True)
        else:
            self.processor = CausalConv1d()

    def forward(self, waveform):
        x = self.feature_encoder(waveform)
        return self.processor(x)


def test_mfcc(batch):
    mfcc = MFCCEncoder(n_mfcc=12, delta=2, hop_time=0.01, sample_rate=16000)
    batch["waveform"].requires_grad_(True)
    print("waveform: ", tuple(batch["waveform"].shape))
    # w = F.pad(batch["waveform"], (mfcc.hop_length // 2, 0))
    # w = F.pad(batch["waveform"], (mfcc.n_fft - 1, 0), mode="replicate")
    w = batch["waveform"]
    m = mfcc(w)
    print("m: ", tuple(m.shape))
    m[..., 500].norm().backward()
    g = batch["waveform"].grad.abs()
    fig, ax = plt.subplots(1, 1)
    ax.plot(g[0])
    plt.show()


if __name__ == "__main__":

    from datasets_turntaking import DialogAudioDM
    import matplotlib.pyplot as plt

    frame_hz = 20  # 50ms
    data_conf = DialogAudioDM.load_config()
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hz=frame_hz,
        vad_bin_times=data_conf["dataset"]["vad_bin_times"],
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=4,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup()

    batch = next(iter(dm.val_dataloader()))

    conf = FeatureEncoder.load_config()
    conf["encoder"]["type"] = "lstm"

    enc = FeatureEncoder(conf)
    x, h = enc(batch["waveform"])
