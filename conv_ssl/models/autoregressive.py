import torch
import torch.nn as nn

from conv_ssl.models.transformer import GPT
from conv_ssl.models.transformer_old import CausalTransformer


class AR(nn.Module):
    """Simplified version of original `CPCAR` module"""

    TYPES = ["gru", "sru", "lstm", "transformer", "gpt"]

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
        assert ar in self.TYPES, 'Please choose ["GRU", "LSTM", "transformer", "gpt"]'

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

        # TODO: input projection if input_dim != dim
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
            # if not transfomer_kwargs["use_pos_emb"] and self.dim != self.input_dim:
            if self.dim != self.input_dim:
                ret = nn.Sequential(
                    nn.Linear(self.input_dim, self.dim), nn.LayerNorm(self.dim), ret
                )
        elif ar == "gpt":
            ret = GPT(
                dim=self.dim,
                dff_k=transfomer_kwargs["dff_k"],
                num_layers=self.num_layers,
                num_heads=transfomer_kwargs["num_heads"],
                activation="GELU",
                dropout=self.dropout,
                use_pos_emb=transfomer_kwargs["use_pos_emb"],  # False -> Alibi
                max_context=transfomer_kwargs["max_context"],
            )

        return ret

    def forward(self, x, attention=False):
        ret = {}
        if self.ar_type == "transformer":
            x = self.ar(x)
            ret["z"] = x
        elif self.ar_type == "gpt":
            x = self.ar(x, attention=attention)
            if attention:
                x, attn = x
                ret["attn"] = attn
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


def _test_ar(config_name):
    from conv_ssl.utils import load_hydra_conf
    from omegaconf import OmegaConf

    conf = load_hydra_conf(config_name=config_name)
    conf = conf["model"]
    print(OmegaConf.to_yaml(conf))
    B = 4
    N = 100
    D = 256
    # Autoregressive
    model = AR(
        input_dim=D,
        dim=conf["ar"]["dim"],
        num_layers=conf["ar"]["num_layers"],
        dropout=conf["ar"]["dropout"],
        ar=conf["ar"]["type"],
        transfomer_kwargs=dict(
            num_heads=conf["ar"]["num_heads"],
            dff_k=conf["ar"]["dff_k"],
            use_pos_emb=conf["ar"]["use_pos_emb"],
            max_context=conf["ar"].get("max_context", None),
            abspos=conf["ar"].get("abspos", None),
            sizeSeq=conf["ar"].get("sizeSeq", None),
        ),
    )
    # print(model)
    x = torch.rand((B, N, D))
    print("x: ", x.shape)
    o = model(x)
    print(o["z"].shape)


if __name__ == "__main__":
    _test_ar("model/discrete")
    _test_ar("model/discrete_20hz")
