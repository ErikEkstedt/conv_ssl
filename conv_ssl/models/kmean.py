from argparse import ArgumentParser
from os.path import dirname
from os import makedirs, cpu_count
import time

import torch
import torch.nn as nn
import einops

from sklearn.cluster import MiniBatchKMeans

# TODO: FAISS-Kmeans
class KMeanFaiss:
    def __init__(self, k):
        self.k = k

    def train(self, X):
        return X


class KmeanSKLearn:
    def __init__(
        self,
        k,
        init="k-means++",
        batch_size=1024,
        tol=0,
        n_init=20,
        reassignment_ratio=0.3,
        max_no_improvement=100,
        random_state=0,
        compute_labels=True,
        init_size=None,
    ):
        self.k = k
        self.model = MiniBatchKMeans(
            n_clusters=k,
            init=init,
            batch_size=batch_size,
            tol=tol,
            n_init=n_init,
            reassignment_ratio=reassignment_ratio,
            max_no_improvement=max_no_improvement,
            random_state=random_state,
            verbose=0,
            compute_labels=compute_labels,
            init_size=init_size,
        )

    def inertia(self, x):
        return -self.model.score(x) / len(x)

    def save_vectors(self, path):
        K = self.get_vectors()
        makedirs(dirname(path), exist_ok=True)
        torch.save(K, path)
        print(f"Saved K-vectors -> {path}")

    def get_vectors(self):
        return torch.from_numpy(self.model.cluster_centers_)

    def train(self, X, verbose=False):
        start_time = time.time()
        self.model.fit(X)
        time_taken = time.time() - start_time
        if verbose:
            m = int(round(time_taken // 60, 2))
            s = round(time_taken - (time_taken // 60) * 60)
            if m > 0:
                print(f"Kmeans for {tuple(X.shape)} took {m} minute {s} seconds")
            else:
                print(f"Kmeans for {tuple(X.shape)} took {s} seconds")

    def partial_fit(self, x, lengths=None):
        if lengths is not None:
            assert x.ndim == 2, "Must provide (B, D) with provided `lengths`"
            x = torch.cat([s[:l] for s, l in zip(x, lengths)])
        self.model = self.model.partial_fit(x)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """argparse arguments for SoSIModel (based on yaml-config)"""
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler="resolve"
        )
        # KMeans arguments
        # ----------------
        # Inspiration: https://github.com/pytorch/fairseq/blob/main/examples/textless_nlp/gslm/speech2unit/clustering/cluster_kmeans.py
        parser.add_argument("-k", "--k", type=int, default=100)
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
        # batch_sizeint, default=1024 Size of the mini batches.
        # For faster compuations, you can set the batch_size
        # greater than 256 * number of cores to enable parallelism on all cores.
        batch_size = 256 * cpu_count()  # greater than?
        parser.add_argument("--k_batch_size", default=batch_size)
        parser.add_argument("--init", default="k-means++")
        parser.add_argument("--tol", default=0, type=float)
        parser.add_argument("--max_no_improvement", default=100, type=int)
        parser.add_argument("--n_init", default=20, type=int)
        parser.add_argument("--reassignment_ratio", default=0.01, type=float)
        return parser


class KMeanEmbedding(nn.Module):
    def __init__(self, k, dim, vectors=None):
        super().__init__()
        self.k = k
        self.dim = dim
        self.emb = nn.Embedding(k, dim)

        if vectors is not None:
            if isinstance(vectors, str):
                self.load_vectors(vectors)
            else:
                self.set_vectors(vectors)

    def load_vectors(self, path):
        vectors = torch.load(path)
        self.set_vectors(vectors)
        print(f"KMEANS: vectors loaded -> {path}")

    def set_vectors(self, v):
        if not isinstance(v, torch.Tensor):
            v = torch.from_numpy(v)
        self.emb.weight.data = v.float()

    def get_embedding(self, idx):
        return self.emb(idx)

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


def test_kmean_sklearn():
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    kmean = KmeanSKLearn(k=10)
    kmean.train(X)
    v = kmean.get_vectors()
    print("v: ", tuple(v.shape))


if __name__ == "__main__":
    pass
