import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from typing import Optional


def prepare_causal_mask(T, device="cpu"):
    mask = torch.tril(torch.ones((T, T), device=device)).view(1, 1, T, T)
    mask.requires_grad_(False)
    return mask


def get_slopes(n):
    """
    * aLiBi slopes for heads.
    * m in Figure 3.
    * Source:
        - https://github.com/ofirpress/attention_with_linear_biases/blob/5b327adc6d131e28b40ba58906b30bb469483519/fairseq/models/transformer.py#L742

    Comments:

    In the paper, we only train models that have 2^a heads for some a. This function has
    some good properties that only occur when the input is a power of 2.
    To maintain that even closest_power_of_2 = 2**math.floor(math.log2(n))
    when the number of heads is not a power of 2, we use this workaround.
    """

    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    # In the paper, we only train models that have 2^a heads for some a. This function has
    # some good properties that only occur when the input is a power of 2. To maintain that even
    # when the number of heads is not a power of 2, we use this workaround.
    if math.log2(n).is_integer():
        slopes = get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        slopes = (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )
    return slopes


def get_relative_bias_matrix(n, num_heads, device="cpu"):
    """Relative Bias matrix for aLiBi embeddings"""
    return torch.arange(n, device=device).view(1, 1, -1).expand(1, num_heads, -1)


class MultiHeadAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float, bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim

        # key, query, value projections for all heads
        self.key = nn.Linear(dim, dim, bias=bias)
        self.query = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)

        # head re-shapers
        self.unstack_heads = Rearrange("b t (h d) -> b h t d", h=self.num_heads)
        self.stack_heads = Rearrange("b h t d -> b t (h d)")

        # regularization
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # output projection
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.scale = 1.0 / math.sqrt(dim)

    def get_scores(self, q: torch.Tensor, k: torch.Tensor):
        """
        Arguments:
            q: (B, heads, T, D)
            k: (B, heads, T, D)

        Return:
            QK:     (B, heads, T, T)
        """
        return torch.einsum("bhid,bhjd->bhij", q, k)

    def mask_scores(self, qk: torch.Tensor, mask=None):
        T = qk.size(-1)
        if mask is None:
            mask = prepare_causal_mask(T, device=qk.device)
        qk = qk.masked_fill(mask == 0, float("-inf"))
        return qk

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, D = Q.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.unstack_heads(self.key(K))  # (B, heads, T, D_head)
        q = self.unstack_heads(self.query(Q))  # (B, heads, T, D_head)
        v = self.unstack_heads(self.value(V))  # (B, heads, T, D_head)

        # QK
        att = self.get_scores(q, k) * self.scale  #  (B, nh, T, T)
        att = self.mask_scores(att, mask)
        att = F.softmax(att, dim=-1)

        # Softmax, dropout, values
        y = self.attn_drop(att) @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # re-assemble all head outputs side by side
        y = self.stack_heads(y)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class MultiHeadAttentionAlibi(MultiHeadAttention):
    def __init__(self, dim: int, num_heads: int, dropout: float, bias: bool = False):
        super().__init__(dim, num_heads, dropout, bias)
        self.m = torch.tensor(get_slopes(num_heads))
        self.m.requires_grad_(False)
        self.mask = None

    def get_alibi_mask(self, T: int, device="cpu"):
        rel_bias_mat = get_relative_bias_matrix(T, self.num_heads, device)
        alibi = rel_bias_mat * self.m.unsqueeze(0).unsqueeze(-1).to(device)

        # Causal mask (standard GPT pask)
        # lower triangle = 1
        # upper triangle = 0
        mask = prepare_causal_mask(T, device)  # (1, 1, T, T)
        # Repeat to get a mask for each head
        mask = mask.repeat(1, self.num_heads, 1, 1)  # (1, num_heads, T, T)
        # fill "future" information with negative infinity
        mask.masked_fill_(mask == 0, float("-inf"))

        # Add causality mask to alibi  (1, num_heads, T, T)
        alibi = alibi.unsqueeze(-2) + mask
        alibi.requires_grad_(False)  # this should not be trained
        return alibi

    def mask_scores(self, qk: torch.Tensor, mask=None):
        T = qk.size(-1)
        if mask is None:
            if self.mask is None or self.mask.shape[-1] < T:
                mask = self.get_alibi_mask(T, device=qk.device)
                self.mask = mask
            else:
                mask = self.mask[..., :T, :T]

        # add aLiBi-mask to qk (see Figure 3.)
        # Addition/translation does not effect softmax (over each row)
        # mentioned in the original representation
        qk = qk + mask.to(qk.device)
        return qk


def _test_alibi():
    """https://github.com/ofirpress/attention_with_linear_biases"""

    import matplotlib.pyplot as plt

    N = 20
    num_heads = 8
    mha = MultiHeadAttentionAlibi(dim=256, num_heads=num_heads, dropout=0)
    mask = mha.get_alibi_mask(N)
    print("mask: ", tuple(mask.shape))

    fig, ax = plt.subplots(num_heads, 1, sharex=True, sharey=True, figsize=(6, 12))
    for h in range(num_heads):
        ax[h].imshow(
            mask[0, h],
            aspect="auto",
            origin="upper",
            interpolation="none",
            vmin=0,
            vmax=10,
            cmap="viridis",
        )
    # plt.pause(0.1)
    plt.show()


if __name__ == "__main__":
    _test_alibi()
