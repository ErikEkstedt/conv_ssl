import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from conv_ssl.models.causal_conv import CausalConv1d
from vector_quantize_pytorch import VectorQuantize


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=2,
        dilation=1,
        stride=1,
        dropout=0.1,
    ) -> None:
        super().__init__()
        self.conv = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self._1x1 = nn.Conv1d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        z = self.conv(x)
        z = einops.rearrange(z, "b c t -> b t c")
        z = self.norm(z)
        z = einops.rearrange(z, "b t c -> b c t")
        z = self.dropout(F.relu(z))
        highway = self._1x1(z)
        skip = F.relu(z + x)
        return skip, highway


class TCN(nn.Module):
    """Source: https://arxiv.org/abs/2107.08248"""

    def __init__(self, in_channels=1, dim=30, num_layers=9) -> None:
        super().__init__()
        dim = 30
        dilations = [1, 2, 4, 6, 16, 32, 64, 128, 256]
        num_layers = len(dilations)
        layer_list = []
        for i in range(num_layers):
            indim = dim
            if i == 0:
                indim = in_channels
            layer_list.append(
                TCNBlock(
                    in_channels=indim,
                    out_channels=dim,
                    kernel_size=2,
                    stride=1,
                    dilation=dilations[i],
                    dropout=0.1,
                )
            )
        self.layers = nn.ModuleList(layer_list)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        skip_matrix = []
        for layer in self.layers:
            x, skip = layer(x)
            skip_matrix.append(skip)
        skip_matrix = torch.stack(skip_matrix, dim=-1)

        # sum skip connections
        skip_matrix = skip_matrix.sum(dim=-1)
        return skip_matrix


class ProsodyEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.code_dim = 10
        self.n_codes = 32
        self.dim = 30
        self.code_decay = 0.8  # the exponential moving average decay, lower means the dictionary will change faster
        self.commitment_weight = 1.0  # the weight on the commitment loss

        self.tcn = TCN(dim=self.dim)
        # self.pooler = nn.MaxPool1d(kernel_size=5)

        self.dim_to_s = nn.Linear(self.dim, 3 * self.code_dim)
        self.vq1 = self._codebook()
        self.vq2 = self._codebook()
        self.vq3 = self._codebook()
        self.s_to_dim = nn.Linear(3 * self.code_dim, self.dim)

    def _codebook(self):
        return VectorQuantize(
            dim=self.code_dim,
            codebook_size=self.n_codes,  # codebook size
            decay=self.code_decay,  # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=self.commitment_weight,  # the weight on the commitment loss
        )

    def extract_codes(self, s):
        s1, s2, s3 = self.dim_to_s(s).chunk(3, dim=1)

        q1, idx1, commit_loss1 = self.vq1(s1)
        q2, idx2, commit_loss2 = self.vq2(s2)
        q3, idx3, commit_loss3 = self.vq3(s3)
        loss = (commit_loss1 + commit_loss2 + commit_loss3) / 3
        q = torch.cat((q1, q2, q3), dim=1)
        d = self.s_to_dim(q)
        return d, loss

    def forward(self, x):
        print("x: ", tuple(x.shape))
        z = self.tcn(x)
        # s = self.pooler(z)
        s = z.max(dim=-1).values
        p, commitment_loss = self.extract_codes(s)
        return p, commitment_loss


if __name__ == "__main__":

    from datasets_turntaking import DialogAudioDM

    frame_hz = 20  # 50ms
    data_conf = DialogAudioDM.load_config()
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        # sample_rate=data_conf["dataset"]["sample_rate"],
        sample_rate=500,
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

    b = TCNBlock(in_channels=1, out_channels=30, kernel_size=2)
    x, skip = b(batch["waveform"].unsqueeze(1))
    print("TCNBlock  x: ", tuple(x.shape))
    print("TCNBlock  skip: ", tuple(skip.shape))

    model = TCN()
    s = model(batch["waveform"])
    print("TCN: s: ", tuple(s.shape))

    pros_encoder = ProsodyEncoder()
    p, closs = pros_encoder(batch["waveform"][:, :500])
    print("Prosody  p: ", tuple(p.shape))
    print("Prosody  commitment loss: ", closs)
