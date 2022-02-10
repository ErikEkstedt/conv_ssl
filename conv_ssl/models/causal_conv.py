import torch


class CausalConv1d(torch.nn.Conv1d):
    """
    Source: https://github.com/pytorch/pytorch/issues/1333
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(
            torch.nn.functional.pad(input, (self.__padding, 0))
        )


if __name__ == "__main__":

    net = CausalConv1d(in_channels=1, out_channels=32, kernel_size=5, dilation=1)

    step = 100
    x = torch.randn(1, 1, 200)
    x.requires_grad_(True)
    o = net(x)
    o[..., step].norm().backward()
    g = x.grad.abs().log()

    pre = 5
    post = 1
    print(g[..., step - pre : step + post + 1])
