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
    """

    def __init__(
        self,
        dim: int = 512,
        ffn_dim: int = 1536,
        num_heads: int = 8,
        ffn_activation: str = "GELU",
        dropout: float = 0.1,
        position_emb: bool = False,
    ):
        super().__init__()
        self.ln_multihead = nn.LayerNorm(dim)
        self.ln_ffnetwork = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout)

        if position_emb:
            self.multihead = MultiHeadAttention(
                dim=dim, num_heads=num_heads, dropout=dropout
            )
        else:
            self.multihead = MultiHeadAttentionAlibi(
                dim=dim, num_heads=num_heads, dropout=dropout
            )
        self.ffnetwork = ffn_block(
            dim, ffn_dim, activation=ffn_activation, dropout=dropout
        )

    def post_layer_norm_forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Not used but kept here for reference"""
        h, attn = self.multihead(Q=x, K=x, V=x, mask=mask)
        h = self.ln_multihead(x + h)
        h = self.ln_ffnetwork(h + self.ffnetwork(h))
        return h, attn

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.ln_multihead(x)
        h, attn = self.multihead(Q=h, K=h, V=h, mask=mask)
        h = x + self.dropout(h)
        h = x + h
        h = h + self.dropout(self.ffnetwork(self.ln_ffnetwork(h)))
        return h, attn


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
    ):
        super().__init__()
        self.dim = dim
        self.dff = int(dim * dff_k)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = dropout
        self.use_pos_emb = use_pos_emb

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

    def forward(self, x, attention=False):
        all_attention = []

        x = self.pos_emb(x)
        for layer in self.layers:
            x, attn = layer(x)
            if attention:
                all_attention.append(attn)

        if attention:
            attn = torch.stack(all_attention, dim=1)
            return x, attn

        return x


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
