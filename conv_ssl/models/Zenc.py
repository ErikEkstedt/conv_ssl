from os.path import join
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import einops

from sklearn.cluster import MiniBatchKMeans
from conv_ssl.utils import load_config, repo_root

DEFAULT_CONFIG = join(repo_root(), "conv_ssl/config/encoder.yaml")


def train_kmeans(X, k=2, verbose=False):
    sk_learn_kmeans = MiniBatchKMeans(n_clusters=k, verbose=0)
    sk_learn_kmeans.fit(X)

    emb = torch.from_numpy(sk_learn_kmeans.cluster_centers_)
    kmeans = KMeans(emb)

    if verbose:
        print(f"K-means training finished (X={X.shape})")

    return kmeans


def load_cpc_pretrained():
    """"""
    checkpoint = torch.load(CHECKPOINTS["cpc"], map_location="cpu")


def plot_time_representations(h, title="time representation", plot=False):
    time, dim = h.size()

    p = h.softmax(dim=-1)
    information = -p.log2()
    entropy = (p * information).sum(-1)

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].set_title(title)
    # Raw vectors
    ax[0].imshow(h.t(), aspect="auto", origin="lower", interpolation="none")
    ax[0].set_ylabel("dim")
    # Information
    information, _ = information.sort(descending=True)
    ax[1].imshow(information.t(), aspect="auto", origin="lower", interpolation="none")
    ax[1].set_ylabel("Information (rank)")
    # Entropy (invariant to sorting)
    ax[2].plot(entropy)
    ax[2].hlines(
        y=math.log2(dim),
        xmin=0,
        xmax=time - 1,
        linestyle="dashed",
        color="k",
        alpha=0.4,
    )
    ax[2].set_ylabel("Entropy")
    ax[2].set_xlabel("Time")
    plt.tight_layout()
    if plot:
        plt.pause(0.01)
    return fig, ax


def plot_time_projection_2d(k, color="k", plot=False):
    """
    k: (T, 2)
    """
    assert k.ndim == 2, "Only 2 dim is implemented"

    time, _ = k.size()
    alpha_min = 0.1
    alpha_max = 0.95
    alpha_range = alpha_max - alpha_min
    alpha_dt = alpha_range / time
    fig, ax = plt.subplots(1, 1)
    # step over time and linearly increase alpha
    for t, (x, y) in enumerate(k):
        alpha = alpha_min + t * alpha_dt
        ax.scatter(x, y, color=color, alpha=alpha)
    plt.tight_layout()
    if plot:
        plt.pause(0.01)

    return fig, ax


def plot_projection_2d(k, pre_dim=None, title=None, color="k", plot=False):
    n, k_dim = k.shape
    fig, ax = plt.subplots(1, 1)
    ax.scatter(k[:, 0], k[:, 1], color=color)
    if title is None:
        if pre_dim is not None:
            ax.set_title(f"K-means {pre_dim} -> {k_dim} (n={n})")
        else:
            ax.set_title(f"K-means")
    else:
        ax.set_title(title)
    ax.set_xlabel("k0")
    ax.set_ylabel("k1")
    plt.tight_layout()
    if plot:
        plt.pause(0.1)
    return fig, ax


class KMeans(nn.Module):
    """
    Now requires training through sk_learn
    """

    def __init__(self, emb, from_dim):
        super().__init__()
        self.k = emb.shape[0]  # k
        self.dim = emb.shape[1]  # dim
        self.emb = nn.Embedding(self.k, self.dim)
        self.emb.weight.data = emb.float()

        self.from_dim = from_dim

    def idx_to_emb(self, idx):
        return self.emb(idx)

    def emb_to_idx(self, x):
        return self(x)

    def plot(self, k, **kwargs):
        return plot_projection_2d(k, **kwargs)

    def forward(self, x):
        b = x.shape[0]

        reshape = False
        if x.ndim == 3:
            reshape = True
            flat_input = einops.rearrange(x, "b t d -> (b t) d")
        else:
            flat_input = x

        # Calculate euclidean distances
        p1 = torch.sum(flat_input ** 2, dim=1, keepdim=True)
        p2 = torch.sum(self.emb.weight ** 2, dim=1)
        p3 = 2 * torch.matmul(flat_input, self.emb.weight.t())
        distances = p1 + p2 - p3

        # Find idx with smallest distance
        enc_idx = distances.argmin(dim=1)
        if reshape:
            enc_idx = einops.rearrange(enc_idx, "(b t) -> b t", b=b)
        return self.emb(enc_idx), enc_idx


class Downsample(nn.Module):
    """
    Downsamples information with minimal learnable contribution
    """

    V = ["mean", "last", "cnn"]

    def __init__(self, dim=512, n_frames=5, type="mean", dimension=-1) -> None:
        super().__init__()
        self.dim = dim
        self.dimension = dimension
        self.n_frames = n_frames
        self.type = type

        assert type in self.V, "Wrong version"

        if type == "cnn":
            self.net = nn.Conv1d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=self.n_frames,
                stride=self.n_frames,
            )
            self._init_prior("sum")
        elif type == "maxpool":
            self.net = nn.MaxPool1d(kernel_size=self.n_frames, stride=self.n_frames)
        elif type == "sum":
            pass
        elif type == "mean":
            self.net = None

    def _init_prior(self, type):
        if type == "sum":
            # identity (+ tiny noise?)
            # averaging?
            torch.nn.init.constant_(self.net.weight, val=1.0)
            torch.nn.init.constant_(self.net.bias, val=0.0)
        elif type == "avg":
            torch.nn.init.constant_(self.net.weight, val=1.0 / self.dim)
            torch.nn.init.constant_(self.net.bias, val=0.0)

    def __repr__(self):
        s = "Downsampler(\n"
        s += f"\ttype: {self.type}\n"
        s += f"\tn_frames: {self.n_frames}\n"
        s += ")"
        return s

    def forward(self, x):
        if self.type == "cnn":
            x = einops.rearrange(x, "b n c -> b c n")
            x = self.net(x)
            x = einops.rearrange(x, "b c n -> b n c")
        elif self.type == "mean":
            x = x.unfold(
                dimension=self.dimension, step=self.n_frames, size=self.n_frames
            ).mean(dim=-1)
        elif self.type == "last":
            x = x.unfold(
                dimension=self.dimension, step=self.n_frames, size=self.n_frames
            )[..., -1]
        return x


class Xenc(nn.Module):
    """Zenc

    * Possible names:
        - Low level perception
        - StaticProjection
    * Automata -> H
    """

    def __init__(self, arg=None):
        super().__init__()
        self.arg = arg

    def plot(self, h, **kwargs):
        """
        h: (t, d)
        """
        return plot_time_representations(h, **kwargs)

    def forward(self, x):
        return x


class Cenc(nn.Module):
    """Cenc

    * Possible names:
        - M (as in the world-model paper)
        - Sequential Lower Tier processor
    * H -> C
    * M(c_t | h_t>= )
    * Time dependent sequence of H
    """

    def __init__(self, arg=None):
        super().__init__()
        self.arg = arg

    def plot(self, c, **kwargs):
        return plot_time_representations(c, **kwargs)

    def forward(self, h):
        return h


class Predictor(nn.Module):
    def __init__(self, arg=None):
        self.arg = arg

    def negative_loss(self, c, hy, hy_negatives):
        return c, hy, hy_negatives

    def forward(self, c, hy):
        """
        c:  (B, T, D)
        hy: (B, T, D)
        """
        sim = torch.nn.functional.cosine_similarity(c, hy, dim=-1)
        return sim


class Yenc(nn.Module):
    def __init__(self, args=None):
        self.args = args

    def plot(self, hy, **kwargs):
        """
        hy:  (T, D)
        """
        return plot_time_representations(hy, **kwargs)

    def plot(self, h, **kwargs):
        """
        h: (t, d)
        """
        return plot_time_representations(h, **kwargs)

    def forward(self, y):
        return y


if __name__ == "__main__":

    import torchaudio

    xenc = Xenc()
    cenc = Cenc()
    predictor = Predictor()
    yenc = Yenc()

    x, sr = torchaudio.load("assets/simple_dialog_start.wav")
    hx = xenc(x)
    cx = cenc(hx)
    hy = yenc(x)
    sim = predictor(cx, hy)
    print("hx: ", tuple(hx.shape))
    print("cx: ", tuple(cx.shape))
    print("hy: ", tuple(hy.shape))
    print("sim: ", tuple(sim.shape))

    # Analysis
    hx_kmean = train_kmeans(X=hx, k=2)
    hy_kmean = train_kmeans(X=hy, k=2)

    """
    Representational Analysis Capabilities
    Functions:
    - K-means
    - 
    """

    t = 25
    k = torch.stack((torch.arange(t), torch.sin(torch.arange(t) / math.pi)), dim=-1)
    _ = plot_projection_2d(k, plot=True)
    _ = plot_time_projection_2d(k, plot=True)
