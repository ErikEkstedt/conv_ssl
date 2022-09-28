import torch
import torch.nn as nn
import math

from typing import Optional, Tuple

from conv_ssl.models.multi_head_attention import (
    MultiHeadAttentionAlibi,
    MultiHeadAttention,
)


class StaticPositionEmbedding(nn.Module):
    def __init__(self, seqlen, dmodel):
        super(StaticPositionEmbedding, self).__init__()
        pos = torch.arange(0.0, seqlen).unsqueeze(1).repeat(1, dmodel)
        dim = torch.arange(0.0, dmodel).unsqueeze(0).repeat(seqlen, 1)
        div = torch.exp(
            -math.log(10000) * (2 * torch.div(dim, 2, rounding_mode="trunc") / dmodel)
        )
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])
        self.register_buffer("pe", pos.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


def ffn_block(
    din: int,
    dff: int,
    activation: str = "GELU",
    dropout: float = 0.0,
    bias: bool = False,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(din, dff, bias=bias),
        getattr(nn, activation)(),
        nn.Dropout(p=dropout),
        nn.Linear(dff, din, bias=bias),
    )


class TransformerLayer(nn.Module):
    """
    Transformer Layer

    Using pre-layer-normalization: https://arxiv.org/pdf/2002.04745.pdf

    Inspiration: https://nn.labml.ai/transformers/models.html
    """

    def __init__(
        self,
        dim: int = 512,
        ffn_dim: int = 1536,
        num_heads: int = 8,
        ffn_activation: str = "GELU",
        dropout: float = 0.1,
        position_emb: bool = False,
        cross_attention: bool = False,
    ):
        super().__init__()
        self.ln_self_attn = nn.LayerNorm(dim)
        self.ln_ffnetwork = nn.LayerNorm(dim)
        if cross_attention:
            self.ln_src_attn = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout)
        self.cross_attention = cross_attention

        MHA = MultiHeadAttention if position_emb else MultiHeadAttentionAlibi

        self.multihead = MHA(dim=dim, num_heads=num_heads, dropout=dropout)
        if cross_attention:
            self.src_multihead = MHA(dim=dim, num_heads=num_heads, dropout=dropout)

        self.ffnetwork = ffn_block(
            dim, ffn_dim, activation=ffn_activation, dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        src: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        z = self.ln_self_attn(x)
        self_attn, attn_self = self.multihead(Q=z, K=z, V=z, mask=mask)

        x = x + self.dropout(self_attn)

        attn_cross = None
        if self.cross_attention and src is not None:
            z = self.ln_src_attn(x)
            src_attn, attn_cross = self.src_multihead(Q=z, K=src, V=src, mask=mask)
            x = x + self.dropout(src_attn)

        x = x + self.dropout(self.ffnetwork(self.ln_ffnetwork(x)))
        return x, attn_self, attn_cross


class GPT(nn.Module):
    def __init__(
        self,
        dim: int,
        dff_k: int = 3,
        num_layers: int = 4,
        num_heads: int = 4,
        activation: str = "GELU",
        dropout: float = 0.1,
        use_pos_emb: bool = False,  # False -> Alibi
        max_context: int = 1024,
        cross_attention: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.dff = int(dim * dff_k)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = dropout
        self.use_pos_emb = use_pos_emb
        self.cross_attention = cross_attention

        if self.use_pos_emb:
            self.max_context = max_context
            self.pos_emb = StaticPositionEmbedding(max_context, self.dim)
        else:
            self.pos_emb = nn.Identity()

        layers = []
        for _ in range(self.num_layers):
            layers.append(
                TransformerLayer(
                    dim=self.dim,
                    ffn_dim=self.dff,
                    num_heads=self.num_heads,
                    ffn_activation=self.activation,
                    dropout=self.dropout,
                    position_emb=self.use_pos_emb,
                    cross_attention=self.cross_attention,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        x: torch.Tensor,
        src: Optional[torch.Tensor] = None,
        attention: bool = False,
    ):
        all_attention = []
        all_attention_cross = []

        x = self.pos_emb(x)
        for layer in self.layers:
            x, attn, cross_attn = layer(x, src=src)
            if attention:
                all_attention.append(attn)
                all_attention_cross.append(cross_attn)

        if attention:
            attn = torch.stack(all_attention, dim=1)
            attn_cross = torch.stack(all_attention_cross, dim=1)
            return x, attn, attn_cross
        return x


# TODO: Add DGSLM Layer class
class GPTStereo(nn.Module):
    def __init__(
        self,
        dim: int,
        dff_k: int = 3,
        num_layers: int = 4,
        num_heads: int = 4,
        activation: str = "GELU",
        dropout: float = 0.1,
        use_pos_emb: bool = False,  # False -> Alibi
        max_context: int = 1024,
        cross_attention: bool = False,
    ):

        self.gpt = GPT(
            dim=dim,
            dff_k=dff_k,
            num_layers=num_layers,
            num_heads=num_heads,
            activation=activation,
            dropout=dropout,
            use_pos_emb=use_pos_emb,
            max_context=max_context,
            cross_attention=cross_attention,
        )
        self.combine = nn.Linear(dim * 2, dim)
        self.ln_combine = nn.LayerNorm(dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, attention: bool = False):

        if attention:
            z1, attn_self1, attn_cross1 = self.gpt(x=x1, src=x2)
            z2, attn_self2, attn_cross2 = self.gpt(x=x2, src=x1)
            z = self.ln_combine(self.combine(torch.cat((z1, z2), dim=-1)))
            return z, {
                "attn_self1": attn_self1,
                "attn_cross1": attn_cross1,
                "attn_self2": attn_self2,
                "attn_cross2": attn_cross2,
            }
        else:
            z1 = self.gpt(x=x1, src=x2)
            z2 = self.gpt(x=x2, src=x1)
            z = self.ln_combine(self.combine(torch.cat((z1, z2), dim=-1)))
        return z


def _test_gpt():
    import matplotlib.pyplot as plt

    model = GPT(dim=256, dff_k=3, num_layers=4, num_heads=8)
    x = torch.rand((4, 20, model.dim))
    with torch.no_grad():
        z, attn = model(x, attention=True)
    print("z: ", tuple(z.shape))
    print("attn: ", tuple(attn.shape))
    b = 0
    fig, ax = plt.subplots(
        model.num_heads, model.num_layers, sharex=True, sharey=True, figsize=(12, 12)
    )
    for n_layer in range(model.num_layers):
        for n_head in range(model.num_heads):
            ax[n_head, n_layer].imshow(
                attn[b, n_layer, n_head],
                aspect="auto",
                origin="upper",
                interpolation="none",
                vmin=0,
                vmax=1,
                cmap="viridis",
            )
            if n_layer == 0:
                ax[n_head, n_layer].set_ylabel(f"Head {n_head}")
            if n_head == 0:
                ax[n_head, n_layer].set_title(f"Layer {n_layer}")
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    _test_gpt()
