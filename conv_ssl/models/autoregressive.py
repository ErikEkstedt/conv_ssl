import torch
import torch.nn as nn


from conv_ssl.models.transformer import CausalTransformer


class AR(nn.Module):
    """Simplified version of original `CPCAR` module"""

    TYPES = ["gru", "sru", "lstm", "transformer"]

    def __init__(
        self,
        input_dim,
        dim,
        num_layers,
        dropout,
        ar="LSTM",
        transfomer_kwargs=None,
        keep_hidden=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.ar_type = ar.lower()
        self.ar = self._ar(ar, transfomer_kwargs)
        self.hidden = None
        self.keep_hidden = keep_hidden

    def _ar(self, ar, transfomer_kwargs):
        ar = ar.lower()
        assert ar in self.TYPES, 'Please choose ["GRU", "LSTM", "transformer"]'

        ret = nn.Identity()
        if ar == "gru":
            ret = nn.GRU(
                self.input_dim, self.dim, num_layers=self.num_layers, batch_first=True
            )
        elif ar == "lstm":
            ret = nn.LSTM(
                self.input_dim, self.dim, num_layers=self.num_layers, batch_first=True
            )
        elif ar == "sru":
            raise NotADirectoryError("SRU not implemented!")
            # ret = SRU(self.input_dim, self.dim, num_layers=self.num_layers)

        elif ar == "transformer":
            ret = CausalTransformer(
                dim=self.dim,
                dff_k=transfomer_kwargs["dff_k"],
                num_layers=self.num_layers,
                num_heads=transfomer_kwargs["num_heads"],
                dropout=self.dropout,
                sizeSeq=transfomer_kwargs["sizeSeq"],
                abspos=transfomer_kwargs["abspos"],
            )
            if not transfomer_kwargs["use_pos_emb"] and self.dim != self.input_dim:
                ret = nn.Sequential(nn.Linear(self.input_dim, self.dim), ret)
            return ret
        return ret

    def forward(self, x):
        ret = {}
        if self.ar_type == "transformer":
            x = self.ar(x)
            ret["z"] = x
        else:
            x, h = self.ar(x)
            ret["z"] = x
            ret["h"] = x
            if self.keep_hidden:
                if isinstance(h, tuple):
                    self.hidden = tuple(x.detach() for x in h)
                else:
                    self.hidden = h.detach()
        return ret


if __name__ == "__main__":

    input_dim = 100
    dim = 256
    num_layers = 2
    # model = AR(
    #     input_dim=dim, dim=dim, num_layers=num_layers, ar="LSTM", keep_hidden=False
    # )
    model = AR(
        input_dim=input_dim,
        dim=dim,
        num_layers=num_layers,
        dropout=0.1,
        ar="transformer",
        transfomer_kwargs={"dff_k": 3, "num_heads": 4, "sizeSeq": 1024, "abspos": True},
        keep_hidden=False,
    )
    print(model)
    B = 4
    N = 100
    x = torch.rand((B, N, input_dim))
    print("x: ", x.shape)
    o = model(x)
    print(o["z"].shape)
