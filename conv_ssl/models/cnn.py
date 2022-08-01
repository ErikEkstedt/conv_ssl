import torch
import torch.nn as nn

from einops.layers.torch import Rearrange


class LayerNorm(nn.Module):
    """
    Extending `nn.LayerNorm` by rearranging input dims to normalize over channel dimension in convnets.

    The original `nn.LayerNorm` + 2 einops Rearrange is faster than custom Norm which calculated values directly on channel...
    """

    def __init__(self, dim: int, rearrange_outputs: bool = True) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.in_rearrange = Rearrange("b d t -> b t d")
        if rearrange_outputs:
            self.out_rearrange = Rearrange("b t d -> b d t")
        else:
            self.out_rearrange = nn.Identity()

    def __repr__(self):
        return str(self.ln)

    def forward(self, x):
        return self.out_rearrange(self.ln(self.in_rearrange(x)))


class CConv1d(nn.Conv1d):
    """source: https://github.com/pytorch/pytorch/issues/1333"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        padding_value=0,
        bias=True,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )

        ks = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        pad_dim1_pre = ks - 1
        pad_dim1_post = 0
        if dilation > 0:
            pad_dim1_pre *= dilation
        pad = (pad_dim1_pre, pad_dim1_post)
        self.pad = nn.ConstantPad1d(padding=pad, value=padding_value)

    def debug_weights(self, type="sum"):
        w = 1.0
        if type == "mean":
            w = 1.0 / self.kernel_size[0]

        elif type == "range":
            k = self.kernel_size[0]
            w = torch.arange(1, k + 1).float().pow(2)
            w = w.repeat(self.out_channels, self.in_channels, 1)
            print("w: ", w.shape)
            self.weight.data = self.weight.data = w
            if self.bias:
                self.bias.data = self.bias.data.fill_(0.0)
            return None

        self.weight.data = self.weight.data.fill_(w)
        if self.bias:
            self.bias.data = self.bias.data.fill_(0.0)

    def forward(self, input):
        return super().forward(self.pad(input))
