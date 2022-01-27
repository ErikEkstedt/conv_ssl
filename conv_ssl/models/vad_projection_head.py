import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from typing import Tuple, Dict

from datasets_turntaking.features.vad import VAD
from conv_ssl.plot_utils import plot_vad_oh
from conv_ssl.utils import find_island_idx_len

import numpy as np


def to_device(batch, device="cuda"):
    new_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            new_batch[k] = v.to(device)
        else:
            new_batch[k] = v
    return new_batch


def eot_moments(vad, n_frames):
    b, n = vad.shape[:2]
    eots_oh = torch.zeros((2, b, n))
    for batch in range(b):
        for speaker in [0, 1]:
            s, d, v = find_island_idx_len(vad[batch, :, speaker])
            active = v == 1
            if active.sum() > 0:
                start = s[active]
                dur = d[active]
                end = start + dur
                keep = torch.where(end < n)[0]
                if len(keep) > 0:
                    start = start[keep]
                    dur = dur[keep]
                    end = end[keep]
                    over_dur = (dur - n_frames) > 0
                    new = torch.where(over_dur)[0]
                    if len(new) > 0:
                        start[new] = end[new] - n_frames
                    for s, e in zip(start, end):
                        eots_oh[speaker, batch, s : e + 1] = 1
    return eots_oh


def plot_debug(
    vad,
    hold_oh,
    shift_oh,
    shift_probs,
    eot_probs,
    next_speaker,
    plot=True,
):
    n = len(vad)
    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(9, 9))
    n_fig = 0
    ###########################################
    # VAD
    _ = plot_vad_oh(vad, ax=ax[n_fig])
    n_fig += 1
    ###########################################
    # Plot HOLD/SHIFT
    _ = plot_vad_oh(vad, ax=ax[n_fig])
    m = hold_oh.shape[0]
    ax[n_fig].fill_between(
        torch.arange(m),
        y1=-torch.ones(m),
        y2=torch.ones(m),
        where=hold_oh,
        color="r",
        alpha=0.07,
        label="HOLD",
    )
    ax[n_fig].fill_between(
        torch.arange(m),
        y1=-torch.ones(m),
        y2=torch.ones(m),
        where=shift_oh,
        color="g",
        alpha=0.07,
        label="SHIFT",
    )
    ax[n_fig].legend(loc="upper left")
    n_fig += 1
    ###########################################
    # Plot Shift/Hold probs
    shift = shift_probs.numpy()
    shift[shift == 0] = np.nan
    ax[n_fig].plot(shift, label="shift %", color="g", linewidth=1.5)
    ax[n_fig].set_xlim([0, n])
    ax[n_fig].set_ylim([0, 1])
    ax[n_fig].legend(loc="upper left")
    n_fig += 1
    eot = eot_probs.numpy()
    eot[eot == 0] = np.nan
    ax[n_fig].plot(eot, label="EoT %", color="r", linewidth=1.5)
    ax[n_fig].set_xlim([0, n])
    ax[n_fig].set_ylim([0, 1])
    ax[n_fig].legend(loc="upper left")
    n_fig += 1
    ###########################################
    # Plot Next Speaker
    ax[n_fig].plot(-next_speaker, linewidth=3, label="Next Speaker")
    ax[n_fig].set_yticks([-0, -1, -2])
    ax[n_fig].set_yticklabels(["A", "B", "Both"])
    ax[n_fig].set_xlim([0, n])
    ax[n_fig].legend(loc="upper left")
    n_fig += 1
    if plot:
        plt.show()
    return fig, ax


class VadProjection(nn.Module):
    def __init__(self, bin_sizes=[20, 40, 60, 80], threshold_ratio=0.5, k=5):
        super().__init__()
        self.n_bins = len(bin_sizes) * 2
        self.n_classes = 2 ** self.n_bins
        self.bin_sizes = bin_sizes
        self.horizon = sum(bin_sizes)
        self.threshold_ratio = threshold_ratio
        self.k = k

        # Onehot-representation vectors
        self.codebook = self.init_codebook()
        self.first_speaker_idx = self.init_first_speaker_mapping()
        self.next_speaker_idx = self.init_next_speaker_idx()
        self.eot_idx = self.init_eot_idx()

        self.eval()
        self.requires_grad_(False)

    def init_codebook(self) -> nn.Module:
        """
        Initializes the codebook for the vad-projection horizon labels.

        Map all vectors of binary digits of length `n_bins` to their corresponding decimal value.

        This allows a VAD future of shape (*, 4, 2) to be flatten to (*, 8) and mapped to a number
        corresponding to the class index.
        """

        def single_idx_to_onehot(idx, d=8):
            assert idx < 2 ** d, "must be possible with {d} binary digits"
            z = torch.zeros(d)
            b = bin(idx).replace("0b", "")
            for i, v in enumerate(b[::-1]):
                z[i] = float(v)
            return z

        def create_code_vectors(n_bins):
            """
            Create a matrix of all one-hot encodings representing a binary sequence of `self.n_bins` places
            Useful for usage in `nn.Embedding` like module.
            """
            n_codes = 2 ** n_bins
            embs = torch.zeros((n_codes, n_bins))
            for i in range(2 ** n_bins):
                embs[i] = single_idx_to_onehot(i, d=n_bins)
            return embs

        codebook = nn.Embedding(
            num_embeddings=self.n_classes, embedding_dim=self.n_bins
        )
        codebook.weight.data = create_code_vectors(self.n_bins)
        codebook.weight.requires_grad_(False)
        return codebook

    def get_single_speaker_vectors(self, n, start=0):
        vectors = []
        for i in range(start, 2 ** n):
            i = bin(i).replace("0b", "").zfill(n)
            tmp = torch.zeros(n)
            for j, val in enumerate(i):
                tmp[j] = float(val)
            vectors.append(tmp)
        return torch.stack(vectors)

    def next_speaker_from_vad_oh(self, x):
        """
        Calculates the next speaker in the label.
        0: speaker 0
        1: speaker 1
        2: equal (both active at same time or no activity)
        Args:
            x:  torch.Tensor: (2, n_bins)
        """

        def single(x):
            first = 2
            for i in range(x.shape[-1]):
                tmp_vad = x[:, i]
                if tmp_vad.sum() == 2:
                    first = 2
                    break
                elif tmp_vad[0] > 0:
                    first = 0
                    break
                elif tmp_vad[1] > 0:
                    first = 1
                    break
            return first

        if x.ndim == 3:  # (N, 2, window)
            first = []
            for xxx in x:
                first.append(single(xxx))
            first = torch.stack(first)
        elif x.ndim == 4:  # (B, N, 2, window)
            first = []
            for batch_x in x:
                tmp_first = []
                for seq_x in batch_x:
                    tmp_first.append(single(seq_x))
                first.append(torch.tensor(tmp_first))
            first = torch.stack(first)
        else:  # (2, window)
            first = single(x)
        return first

    def init_first_speaker_mapping(self, both_is_valid_target=True) -> torch.Tensor:
        """
        Map all classes and corresponding one-hot representation to a small set of labels
        which encodes which speaker the first non-zero activity belongs to.

        Used in order to take turns based on future window prediction and whether it would
        be considered a Shift or a Hold.
        """

        # 0:A, 1:B, 2:equal
        ns2idx = {0: [], 1: [], 2: []}  # next-speaker 2 index

        idx = torch.arange(self.n_classes)  # get all class idx
        vad_labels_oh = self.idx_to_onehot(idx)  # turn to vad onehot
        for i, vl in enumerate(vad_labels_oh):  # check who is the first speaker
            n = self.next_speaker_from_vad_oh(vl)
            ns2idx[n].append(i)

        # List -> tensors
        for i, v in ns2idx.items():
            ns2idx[i] = torch.tensor(v)

        # add both as valid idx for each speaker
        if both_is_valid_target:
            first_speaker_a = torch.cat((ns2idx[0], ns2idx[2]))
            first_speaker_b = torch.cat((ns2idx[1], ns2idx[2]))
            first_speaker = torch.stack(
                (first_speaker_a, first_speaker_b)
            )  # (2, M+M_b)
        else:
            first_speaker = torch.stack((ns2idx[0], ns2idx[1]))  # (2, M+M_b)

        return first_speaker.long()

    def init_next_speaker_idx(self) -> Dict:
        """
        Change speaker idx

        Collects all indices for speaker change. Read "the next speaker is 1 or 0"

        Given that we know who the current speaker (in mututal silences the previous speaker is used) we have
        all indices corresponding to a speaker change.

        the strict indices corresponds to activity where only a single speaker is active (including all silent).

        The non-strict indices corresponds to non-zero activity (from the left i.e. close to the current moment) from
        the 'other' speaker but where the activity is always overtaken by the corresponding speaker.

        i.e. we know that the current/previous speaker is 0 then we can get all indicies corresponding
        to a speaker shift -> speaker 1 by:

        ```python
            current_speaker = 0
            next_speaker = 1
            idx_corresponding_to_shifts = self.next_speaker['all'][next_speaker]
        ```

        Or if we are in a mutual silence and only want to check the idx where only speaker 1 is the next speaker

        ```python
            next_speaker = 1
            idx_corresponding_to_shifts = self.next_speaker['strict'][next_speaker]
        ```
        """
        # Strict
        vec = self.get_single_speaker_vectors(n=len(self.bin_sizes))
        z = torch.zeros_like(vec)
        b = torch.stack((z, vec), dim=1)
        a = torch.stack((vec, z), dim=1)
        aidx = self.onehot_to_idx(a).sort().values
        bidx = self.onehot_to_idx(b).sort().values
        next_speaker = {"strict": torch.stack((aidx, bidx))}
        # strict 1
        ii = 1
        for ii in range(1, len(self.bin_sizes)):
            vec = self.get_single_speaker_vectors(n=len(self.bin_sizes) - ii, start=1)
            starts = self.get_single_speaker_vectors(n=ii)
            s = starts.repeat(vec.shape[0], 1)
            v = vec.repeat(starts.shape[0], 1)
            v = torch.cat((s, v), dim=-1)
            z = torch.zeros_like(v)
            z[:, :ii] = 1
            tmp_a = torch.stack((v, z), dim=1)
            tmp_b = torch.stack((z, v), dim=1)
            tmp_a_idx = self.onehot_to_idx(tmp_a).sort().values
            tmp_b_idx = self.onehot_to_idx(tmp_b).sort().values
            next_speaker[f"non_strict_{ii}"] = torch.stack((tmp_a_idx, tmp_b_idx))
        next_speaker_a = (
            torch.cat([a[0] for _, a in next_speaker.items()]).sort().values
        )
        next_speaker_b = (
            torch.cat([a[1] for _, a in next_speaker.items()]).sort().values
        )
        next_speaker["all"] = torch.stack((next_speaker_a, next_speaker_b))
        return next_speaker

    def init_eot_idx(self):
        """
        End of Turn indices

        given that we know the speaker of the current turn these indices corresponds
        to projections where the current user activity ends inside of the projection window.

        We know speaker 0 is currently active than we have all indices corresponding to speaker 0
        is not active at the end of the projection horizon:

        ```python
            current_speaker = 0
            idx_corresponding_to_eot = self.eot_idx[current_speaker]
        ```

        """
        # n = len(self.bin_sizes)
        # eot_oh = []
        # all_possible_vec = self.get_single_speaker_vectors(n=n)
        # n = len(self.bin_sizes)
        # for i in range(n):
        #     tmp_eot_single = torch.zeros_like(all_possible_vec)
        #     tmp_eot_single[:, :i] = 1
        #     tmp_eot = torch.stack((tmp_eot_single, all_possible_vec), dim=1)
        #     eot_oh.append(tmp_eot)
        # eot_oh = torch.cat(eot_oh)
        #
        # # Symmetric for other channel
        # eot_other = torch.stack((eot_oh[:, 1], eot_oh[:, 0]), dim=1)
        # eot_oh = torch.stack((eot_oh, eot_other))
        # eot_idx = self.onehot_to_idx(eot_oh)

        n = len(self.bin_sizes)
        all_possible_vec = self.get_single_speaker_vectors(n)
        ones = torch.ones(
            (
                all_possible_vec.shape[0],
                n,
            )
        )
        a = torch.stack((ones, all_possible_vec), dim=1)
        b = torch.stack((all_possible_vec, ones), dim=1)
        a_idx = self.onehot_to_idx(a)
        b_idx = self.onehot_to_idx(b)
        class_idx = torch.arange(self.n_classes)
        where_not = (class_idx.unsqueeze(-1) == a_idx).sum(dim=-1)
        where = torch.logical_not(where_not)
        a_eot = class_idx[where]
        where_not = (class_idx.unsqueeze(-1) == b_idx).sum(dim=-1)
        where = torch.logical_not(where_not)
        b_eot = class_idx[where]
        eot_idx = torch.stack((a_eot, b_eot))

        # Eot
        # import itertools
        #
        # n = len(self.bin_sizes)
        # # nn = n - 1
        # nn = n
        # m = []
        # for i in range(1, nn):
        #     b = torch.zeros((nn,))
        #     b[:i] = 1
        #     m.append(b.long().tolist())
        # curr_vec = []
        # for mm in m:
        #     for a in list(itertools.permutations(mm, r=nn)):
        #         # a = [1] + list(a)
        #         if a not in curr_vec:
        #             curr_vec.append(a)
        # curr_vec = torch.tensor(curr_vec)
        # all_possible_vec = self.get_single_speaker_vectors(n=n)
        # eot_oh_a = []
        # for cv in curr_vec:
        #     for pos in all_possible_vec:
        #         tmp = torch.stack((cv, pos))
        #         eot_oh_a.append(tmp)
        # eot_oh_a = torch.stack(eot_oh_a)
        # eot_oh_b = torch.stack((eot_oh_a[:, 1], eot_oh_a[:, 0]), dim=1)
        # eot_oh = torch.stack((eot_oh_a, eot_oh_b))
        # eot_idx = self.onehot_to_idx(eot_oh)
        # print("eot_idx: ", tuple(eot_idx.shape))
        # print("eot_oh: ", tuple(eot_idx.shape))

        return eot_idx

    def horizon_to_onehot(self, vad_projections):
        """
        Iterate over the bin boundaries and sum the activity
        for each channel/speaker.
        divide by the number of frames to get activity ratio.
        If ratio is greater than or equal to the threshold_ratio
        the bin is considered active
        """
        start = 0
        v_bins = []
        for b in self.bin_sizes:
            end = start + b
            m = vad_projections[..., start:end].sum(dim=-1) / b
            m = (m >= self.threshold_ratio).float()
            v_bins.append(m)
            start = end
        v_bins = torch.stack(v_bins, dim=-1)  # (*, t, c, n_bins)
        # Treat the 2-channel activity as a single binary sequence
        v_bins = v_bins.flatten(-2)  # (*, t, c, n_bins) -> (*, t, (c n_bins))
        return einops.rearrange(v_bins, "... (c d) -> ... c d", c=2)

    def vad_to_label_oh(self, vad) -> torch.Tensor:
        """
        Given a sequence of binary VAD information (two channels) we extract a prediction horizon
        (frame length = the sum of all bin_sizes).

        ! WARNING ! VAD is expected to be shifted one step to get the 'next frame horizon'

        ```python
        # vad: (B, N, 2)
        # DO THIS
        vad_label_idx = VadProjection.vad_to_idx(vad[:, 1:])
        ```

        Arguments:
            vad:        torch.Tensor, (b, n, c) or (n, c)
        """
        # (b, n, c) -> (b, N, c, M), M=horizon window size, N=valid frames
        vad_projections = vad.unfold(dimension=-2, size=sum(self.bin_sizes), step=1)

        # (b, N, c, M) -> (B, N, 2, len(self.bin_sizes))
        v_bins = self.horizon_to_onehot(vad_projections)
        return v_bins

    def vad_to_label_idx(self, vad) -> torch.Tensor:
        """
        Given a sequence of binary VAD information (two channels) we extract a prediction horizon
        (frame length = the sum of all bin_sizes).

        ! WARNING ! VAD is shifted one step to get the 'next frame horizon'

        ```python
        # vad: (B, N, 2)
        # DONT DO THIS
        vad_label_idx = VadProjection.vad_to_idx(vad[:, 1:])
        ```

        Arguments:
            vad:        torch.Tensor, (b, n, c) or (n, c)

        Returns:
            classes:    torch.Tensor (b, t) or (t,)
        """
        v_bins = self.vad_to_label_oh(vad)
        return self.onehot_to_idx(v_bins)

    def onehot_to_idx(self, x) -> torch.Tensor:
        """
        The inverse of the 'forward' function.

        Arguments:
            x:          torch.Tensor (*, 2, 4)

        Inspiration for distance calculation:
            https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/vector_quantize_pytorch.py
        """
        assert x.shape[-2:] == (2, self.n_bins // 2)

        # compare with codebook and get closest idx
        shape = x.shape
        flatten = einops.rearrange(
            x, "... c bpp -> (...) (c bpp)", c=2, bpp=self.n_bins // 2
        )
        embed = self.codebook.weight.t()

        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = dist.max(dim=-1).indices
        embed_ind = embed_ind.view(*shape[:-2])
        return embed_ind

    def idx_to_onehot(self, idx):
        v = self.codebook(idx)
        return einops.rearrange(v, "... (c b) -> ... c b", c=2)

    def topk_hold_shift(self, probs, w_ab, w_ba, w_aa, w_bb):
        def get_topk_score(topk, label):
            a = topk.unsqueeze(-1) == label  # (N, k, label_size)
            a_correct = torch.clamp_max(a.sum(dim=-1).cumsum(dim=-1), 1)
            return a_correct

        p_ab = probs[w_ab]
        p_ba = probs[w_ba]
        p_aa = probs[w_aa]
        p_bb = probs[w_bb]
        a_next = self.first_speaker_idx[0]
        b_next = self.first_speaker_idx[1]

        f1_pred, f1_label = [], []

        data = {}
        score = {}

        # HOld
        hold_topk = torch.zeros(
            (*probs.shape[:-1], self.k), dtype=torch.long, device=probs.device
        )
        hold_topk_probs = torch.zeros_like(hold_topk).float()
        hold_topk_correct = torch.zeros_like(hold_topk)
        # Shift
        shift_topk = torch.zeros_like(hold_topk)
        shift_topk_probs = torch.zeros_like(hold_topk).float()
        shift_topk_correct = torch.zeros_like(hold_topk)

        # A->B, B->A
        # SHIFT
        ab_prob, ab_topk = p_ab.topk(self.k)
        ba_prob, ba_topk = p_ba.topk(self.k)
        topk_ab_correct = get_topk_score(ab_topk, b_next)
        topk_ba_correct = get_topk_score(ba_topk, a_next)
        shift_correct = torch.cat((topk_ab_correct, topk_ba_correct))
        n_shift = topk_ab_correct.shape[0] + topk_ba_correct.shape[0]
        shift_topk[w_ab] = ab_topk
        shift_topk[w_ba] = ba_topk
        shift_topk_probs[w_ab] = ab_prob
        shift_topk_probs[w_ba] = ba_prob
        shift_topk_correct[w_ab] = topk_ab_correct
        shift_topk_correct[w_ba] = topk_ba_correct

        # A->A, B->B
        # HOLD
        aa_prob, aa_topk = p_aa.topk(self.k)
        bb_prob, bb_topk = p_bb.topk(self.k)
        topk_aa_correct = get_topk_score(aa_topk, a_next)
        topk_bb_correct = get_topk_score(bb_topk, b_next)
        hold_correct = torch.cat((topk_aa_correct, topk_bb_correct))
        n_hold = topk_aa_correct.shape[0] + topk_bb_correct.shape[0]
        hold_topk[w_aa] = aa_topk
        hold_topk[w_bb] = bb_topk
        hold_topk_probs[w_aa] = aa_prob
        hold_topk_probs[w_bb] = bb_prob
        hold_topk_correct[w_aa] = topk_aa_correct
        hold_topk_correct[w_bb] = topk_bb_correct

        score["shift"] = {
            "topk": shift_correct.sum(dim=0),
            "n": n_shift,
        }
        score["hold"] = {
            "topk": hold_correct.sum(dim=0),
            "n": n_hold,
        }

        # f1 preds for shift -> 1 is correct 0 is fail
        # topk here but probably only 'greedy' is going to be used
        # f1_pred: 0 -> means predicting hold, 1 -> means predicting shift
        f1_pred.append(shift_correct)  # shift predictions kept as are
        f1_label.append(torch.ones_like(shift_correct))  # shift labels = 1
        f1_pred.append(
            torch.logical_not(hold_correct)
        )  # hold predictions are inverted (0->hold pred)
        f1_label.append(torch.zeros_like(hold_correct))  # hold labels = 0
        f1_pred = torch.cat(f1_pred)
        f1_label = torch.cat(f1_label)
        score["f1_pred"] = f1_pred
        score["f1_label"] = f1_label

        data = {}
        data["hold"] = hold_topk
        data["hold_probs"] = hold_topk_probs
        data["hold_correct"] = hold_topk_correct
        data["shift"] = shift_topk
        data["shift_probs"] = shift_topk_probs
        data["shift_correct"] = shift_topk_correct
        return data, score

    def probs_hold_shift(self, probs, w_ab, w_ba, w_aa, w_bb):
        shift_probs = torch.zeros(
            probs.shape[:-1], dtype=torch.float, device=probs.device
        )
        hold_probs = torch.zeros_like(shift_probs)

        a_next = self.first_speaker_idx[0]
        b_next = self.first_speaker_idx[1]

        # Sum together appropriate probabilities
        p_ab = probs[w_ab][..., b_next].sum(dim=-1)
        p_ba = probs[w_ba][..., a_next].sum(dim=-1)
        p_aa = probs[w_aa][..., a_next].sum(dim=-1)
        p_bb = probs[w_bb][..., b_next].sum(dim=-1)

        score = {}
        data = {}

        shift_probs[w_ab] = p_ab
        shift_probs[w_ba] = p_ba
        n_shift = p_ab.shape[0] + p_ba.shape[0]
        score["shift"] = {"prob": torch.cat((p_ab, p_ba)).mean(), "n": n_shift}
        data["shift"] = shift_probs

        hold_probs[w_aa] = p_aa
        hold_probs[w_bb] = p_bb
        n_hold = p_aa.shape[0] + p_bb.shape[0]
        score["hold"] = {"prob": torch.cat((p_aa, p_bb)).mean(), "n": n_hold}
        data["hold"] = hold_probs
        return data, score

    def get_hold_shift(self, probs, vad, metrics=["probs", "topk"]):
        hold_oh, shift_oh = VAD.get_hold_shift_onehot(vad)
        last_speaker = VAD.get_last_speaker(vad)

        # get probabilities associated with shift/hold
        a_last = last_speaker == 0
        b_last = last_speaker == 1

        # Extract onehot idx:  A->A, A->B, B->A, B->B
        w_ab = torch.where(torch.logical_and(shift_oh, b_last))
        w_ba = torch.where(torch.logical_and(shift_oh, a_last))
        w_aa = torch.where(torch.logical_and(hold_oh, a_last))
        w_bb = torch.where(torch.logical_and(hold_oh, b_last))

        data, score = {}, {}
        if "topk" in metrics:
            data_, score_ = self.topk_hold_shift(probs, w_ab, w_ba, w_aa, w_bb)
            data["topk"] = data_
            score["topk"] = score_

        if "probs" in metrics:
            data_, score_ = self.probs_hold_shift(probs, w_ab, w_ba, w_aa, w_bb)
            data["probs"] = data_
            score["probs"] = score_
        return data, score

    def get_change_probs(self, probs, a_current, b_current):
        # Speaker Change Probs
        change_prob = torch.zeros(
            tuple(probs.shape[:-1]), dtype=torch.float, device=probs.device
        )
        w_a = torch.where(a_current)
        w_b = torch.where(b_current)
        an = self.next_speaker_idx["all"][0]
        bn = self.next_speaker_idx["all"][1]
        change_prob[w_a] = probs[w_a][..., bn].sum(dim=-1)
        change_prob[w_b] = probs[w_b][..., an].sum(dim=-1)
        return change_prob

    def get_eot_probs(self, probs, a_current, b_current, vad):
        eot_probs = torch.zeros(
            probs.shape[:-1], dtype=torch.float, device=probs.device
        )
        eot_probs[a_current] = probs[a_current][..., self.eot_idx[0]].sum(dim=-1)
        eot_probs[b_current] = probs[b_current][..., self.eot_idx[1]].sum(dim=-1)

        data = {}
        score = {}
        data["probs"] = eot_probs
        # eot_cutoff = 0.5
        # # where is probability mass larger than eot_cuttoff
        # is_eot = eot_probs >= eot_cutoff
        # # find eot-moments
        # eot_oh = eot_moments(vad, n_frames=sum(self.bin_sizes))
        # w_wot = torch.where(eot_oh)
        return data, score

    def forward(self, logits, vad, metrics=["topk", "probs"]):
        probs = logits.softmax(dim=-1)
        ds = VAD.vad_to_dialog_vad_states(vad)
        a_current = ds == 0  # current speaker a
        b_current = ds == 3  # current speaker a

        # Device
        self.first_speaker_idx = self.first_speaker_idx.to(logits.device)
        self.next_speaker_idx["all"] = self.next_speaker_idx["all"].to(logits.device)
        self.eot_idx = self.eot_idx.to(logits.device)

        # probs
        data, score = self.get_hold_shift(
            probs, vad, metrics=metrics
        )  # shift_probs, hold_probs

        if "probs" in metrics:
            data["probs"]["change"] = self.get_change_probs(probs, a_current, b_current)
            data_, score_ = self.get_eot_probs(probs, a_current, b_current, vad)
            data["probs"]["eot"] = data_["probs"]
            # data["probs"]["eot"] = self.get_eot_probs(probs, a_current, b_current, vad)
        return data, score


def plot_vad_label(
    vad_label_oh,
    frames=[10, 20, 30, 40],
    colors=["b", "orange"],
    yticks=["B", "A"],
    ylabel=None,
    label=(None, None),
    legend_loc="best",
    alpha=0.9,
    ax=None,
    figsize=(6, 4),
    plot=False,
):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    frame_starts = torch.tensor(frames).cumsum(0)[:-1]
    x = torch.arange(sum(frames))
    expanded = []
    for i, f in enumerate(frames):
        expanded.append(vad_label_oh[:, i].repeat(f, 1))
    expanded = torch.cat(expanded)

    if expanded[:, 0].sum() > 0:
        ax.fill_between(
            x,
            y1=0,
            y2=expanded[:, 0],
            step="pre",
            alpha=alpha,
            color=colors[0],
            label=label[0],
        )
    if expanded[:, 1].sum() > 0:
        ax.fill_between(
            x,
            y1=-expanded[:, 1],
            y2=0,
            step="pre",
            alpha=alpha,
            color=colors[1],
            label=label[1],
        )
    ax.set_ylim([-1.05, 1.05])
    ax.set_xlim([0, len(x) - 1])
    ax.set_xticks([])
    ax.hlines(y=0, xmin=0, xmax=len(x), color="k", linestyle="dashed", linewidth=1)
    ax.vlines(x=frame_starts - 1, ymin=-1, ymax=1, color="k", linewidth=2)

    if label[0] is not None:
        ax.legend(loc=legend_loc)

    if yticks is None:
        ax.set_yticks([])
    else:
        ax.set_yticks([-0.5, 0.5])
        ax.set_yticklabels(yticks)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if plot:
        plt.tight_layout()
        plt.pause(0.01)
    return fig, ax


def plot_vad_projection(
    vad,
    vad_history,
    shift_probs,
    hold_probs,
    change_probs,
    eot_probs,
    probs,
    hold_oh,
    shift_oh,
    eot_pos_oh,
    eot_neg_oh,
    eot_cutoff=0.6,
    figsize=(9, 12),
    plot=True,
):
    """"""

    linewidth = 2
    vad_alpha = 0.2

    n = vad.shape[0]
    fig, ax = plt.subplots(8, 1, sharex=True, figsize=figsize)

    n_fig = 0
    _ = plot_vad_oh(vad, ax[0], legend_loc="upper right", yticks=[])
    ax[n_fig].fill_between(
        torch.arange(n),
        y1=-torch.ones(n),
        y2=torch.ones(n),
        where=hold_oh,
        color="r",
        alpha=0.1,
        label="HOLD",
    )
    ax[n_fig].fill_between(
        torch.arange(n),
        y1=-torch.ones(n),
        y2=torch.ones(n),
        where=shift_oh,
        color="g",
        alpha=0.1,
        label="SHIFT",
    )
    ax[n_fig].legend(loc="upper left")
    ax[n_fig].set_ylabel("Vad")
    ax[n_fig].set_yticks([])
    n_fig += 1
    ########################################################
    # Vad history
    _ = ax[n_fig].imshow(
        vad_history[..., 0].t(),
        aspect="auto",
        interpolation="none",
        origin="lower",
        extent=(0, vad_history.shape[0], 0, vad_history.shape[1]),
    )
    ax[n_fig].set_ylabel("Vad history")
    ax[n_fig].set_yticks([])
    n_fig += 1
    ########################################################
    # Shift/HOLD Probs
    _ = plot_vad_oh(
        vad, ax[n_fig], alpha=vad_alpha, legend_loc="upper right", yticks=[]
    )
    ax[n_fig].fill_between(
        torch.arange(n),
        y1=-torch.ones(n),
        y2=torch.ones(n),
        where=shift_oh,
        color="g",
        alpha=0.1,
        label="Shift",
    )
    ax[n_fig].fill_between(
        torch.arange(n),
        y1=-torch.ones(n),
        y2=torch.ones(n),
        where=hold_oh,
        color="r",
        alpha=0.1,
        label="Hold",
    )
    # ax[n_fig].legend(loc="upper right")
    ax[n_fig].set_ylabel("Shift/Hold")
    ax[n_fig].set_yticks([])
    ax1 = ax[n_fig].twinx()
    ax1.plot(shift_probs, color="g", label="shift %", linewidth=linewidth)
    ax1.plot(hold_probs, color="r", label="hold %", linewidth=linewidth)
    ax1.legend(loc="upper left")
    ax1.set_ylim([0, 1])
    ax1.set_yticks([])
    n_fig += 1
    ########################################################
    # Speaker Change Probs
    _ = plot_vad_oh(
        vad, ax[n_fig], alpha=vad_alpha, legend_loc="upper right", yticks=[]
    )
    ax[n_fig].set_ylabel("Change")
    ax[n_fig].set_yticks([])
    ax2 = ax[n_fig].twinx()
    ax2.plot(change_probs, color="k", label="change %", linewidth=linewidth)
    ax2.plot(shift_probs, color="g", label="shift %", linewidth=linewidth)
    non_hold = 1 - hold_probs
    non_hold[non_hold == 1] = 0
    ax2.plot(non_hold, color="darkred", label="not holds %", linewidth=linewidth)
    ax2.legend(loc="upper left")
    ax2.set_ylim([0, 1])
    ax2.set_yticks([])
    n_fig += 1

    ########################################################
    # EOT
    _ = plot_vad_oh(
        vad, ax[n_fig], alpha=vad_alpha, legend_loc="upper right", yticks=[]
    )
    ax[n_fig].set_ylabel("EOT")
    ax3 = ax[n_fig].twinx()
    ax3.plot(eot_probs, color="k", label="EOT %", linewidth=linewidth)
    ax3.set_ylim([0, 1])
    ax3.set_yticks([])
    ax3.legend(loc="upper left")
    n_fig += 1

    ########################################################
    # EOT CUtoff
    ax[n_fig].set_ylabel("EOT")
    w_eot_tp = torch.logical_and(eot_probs >= eot_cutoff, eot_pos_oh)
    w_eot_fn = torch.logical_and(eot_probs < eot_cutoff, eot_pos_oh)
    w_eot_tn = torch.logical_and(eot_probs < eot_cutoff, torch.logical_not(eot_pos_oh))
    w_eot_fp = torch.logical_and(eot_probs >= eot_cutoff, torch.logical_not(eot_pos_oh))
    ax[n_fig].cla()
    ax[n_fig].hlines(
        y=eot_cutoff, xmin=0, xmax=n, color="k", linestyle="dashed", linewidth=0.5
    )
    ax[n_fig].fill_between(
        torch.arange(n),
        y1=torch.zeros(n),
        y2=torch.ones(n),
        where=eot_pos_oh,
        color="b",
        alpha=0.1,
        label="EOT True",
    )
    ax[n_fig].fill_between(
        torch.arange(n),
        y1=torch.zeros(n),
        y2=torch.ones(n),
        where=eot_neg_oh,
        color="k",
        alpha=0.05,
        label="EOT False",
    )
    ax[n_fig].fill_between(
        torch.arange(n),
        y1=torch.zeros(n),
        y2=eot_probs,
        where=w_eot_tp,
        color="g",
        alpha=0.5,
        label="Correct",
    )
    ax[n_fig].fill_between(
        torch.arange(n),
        y1=torch.zeros(n),
        y2=eot_probs,
        where=w_eot_fn,
        color="r",
        alpha=0.5,
        label="Wrong",
    )
    ax[n_fig].fill_between(
        torch.arange(n),
        y1=torch.zeros(n),
        y2=eot_probs,
        where=w_eot_tn,
        color="g",
        alpha=0.5,
    )
    ax[n_fig].fill_between(
        torch.arange(n),
        y1=torch.zeros(n),
        y2=eot_probs,
        where=w_eot_fp,
        color="r",
        alpha=0.5,
    )
    ax[n_fig].legend(loc="upper left")
    ax[n_fig].set_ylabel("EOT %")
    ax[n_fig].set_yticks([])
    n_fig += 1

    #####################################################3
    # Topk probs
    topk_probs, _ = probs.topk(5)
    ns, nk = topk_probs.shape
    _ = ax[n_fig].imshow(
        topk_probs.t(),
        aspect="auto",
        interpolation="none",
        extent=(0, ns, nk, 0),
    )
    ax[n_fig].set_yticks([])
    ax[n_fig].set_ylabel("Topk %")
    n_fig += 1
    #####################################################3
    # Entropy
    # get max entropy
    n_classes = probs.shape[-1]
    p_uni = torch.ones((n_classes)) / n_classes
    max_entropy = -(p_uni * p_uni.log()).sum()
    entropy = torch.distributions.Categorical(probs=probs).entropy()
    rel_ent = entropy / max_entropy
    ax[n_fig].cla()
    ax[n_fig].plot(rel_ent, color="k", linewidth=2)
    ax[n_fig].set_yticks([])
    ax[n_fig].set_ylabel("entropy (rel)")
    n_fig += 1

    if plot:
        plt.pause(0.1)

    return fig, ax


def speaker_change_vector_debug():
    vad_projection_head = VadProjection(bin_sizes)
    next_a_idx = vad_projection_head.next_speaker_idx["all"][1]
    next_speaker_a_oh = vad_projection_head.idx_to_onehot(next_a_idx)
    first_speaker_a = vad_projection_head.first_speaker_idx[0]
    first_speaker_a_oh = vad_projection_head.idx_to_onehot(first_speaker_a)

    ####################################################################
    # Plot all first speaker projections
    n_rows, n_cols = 10, 9
    fig, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(21, 21))
    j = 0
    for row in range(n_rows):
        for col in range(n_cols):
            plot_vad_label(first_speaker_a_oh[j], ax=ax[row, col])
            j += 1
            if j >= first_speaker_a_oh.shape[0]:
                break
        if j >= first_speaker_a_oh.shape[0]:
            break
    plt.tight_layout()
    plt.show()

    plt.close("all")

    ####################################################################
    # Plot all next speaker projections
    fig, ax = plt.subplots(10, 5, sharex=True, sharey=True, figsize=(21, 21))
    j = 0
    for row in range(10):
        for col in range(5):
            plot_vad_label(next_speaker_a_oh[j], ax=ax[row, col])
            j += 1
    plt.tight_layout()
    plt.show()

    plt.close("all")

    ####################################################################
    # EOT
    # Plot all EoT projections
    eot_idx = vad_projection_head.eot_idx
    eot_oh = vad_projection_head(eot_idx)
    print("eot_oh: ", tuple(eot_oh.shape))
    print("eot_idx: ", tuple(eot_idx.shape))
    print("eot_idx unique: ", tuple(eot_idx[0].unique().shape))
    print("eot_idx unique: ", tuple(eot_idx[1].unique().shape))
    n_rows, n_cols = 8, 8
    fig, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(21, 21))
    j = 0
    for row in range(n_rows):
        for col in range(n_cols):
            plot_vad_oh(eot_oh[0, j].permute(1, 0), ax=ax[row, col])
            j += 1
    ax[0, 0].set_xlim([0.5, 3.5])
    ax[0, 0].set_xticks([])
    plt.tight_layout()
    plt.show()
    plt.close("all")

    n_rows, n_cols = 9, 11
    fig, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(21, 21))
    j = 0
    for row in range(n_rows):
        for col in range(n_cols):
            plot_vad_oh(eot_oh[j].permute(1, 0), ax=ax[row, col])
            j += 1
            if j >= eot_oh.shape[0]:
                break
        if j >= eot_oh.shape[0]:
            break
    ax[0, 0].set_xlim([0.5, 3.5])
    ax[0, 0].set_xticks([])
    plt.tight_layout()
    plt.show()
    plt.close("all")


def valid():
    def get_all_permutations(n, start=0):
        vectors = []
        for i in range(start, 2 ** n):
            i = bin(i).replace("0b", "").zfill(n)
            tmp = torch.zeros(n)
            for j, val in enumerate(i):
                tmp[j] = float(val)
            vectors.append(tmp)
        return torch.stack(vectors)

    def create_next_active_vectors(n=4, min_active=2):
        base = torch.zeros(n)
        if min_active > 0:
            base[-min_active:] = 1
        permutable = n - min_active
        if permutable > 0:
            perms = get_all_permutations(permutable)
            base = base.repeat(perms.shape[0], 1)
            base[:, :permutable] = perms
        return base

    def create_prev_active_end(n=4, max_active=2):
        v = create_next_active_vectors(n, min_active=n - max_active)
        return torch.logical_not(v).long()

    def combine_speakers(x1, x2, mirror=False):
        if x1.ndim == 1:
            x1 = x1.unsqueeze(0)
        if x2.ndim == 1:
            x2 = x2.unsqueeze(0)
        vad = []
        for a in x1:
            for b in x2:
                vad.append(torch.stack((a, b), dim=0))

        vad = torch.stack(vad)
        if mirror:
            vad = torch.stack((vad, torch.stack((vad[:, 1], vad[:, 0]), dim=1)))

        return vad

    #######################################################################
    def get_turn_shift(n=4, next_min_active=2, prev_max_active=2):
        """
        Used in 'mutual silence' segments in the dialog.
        #     B       ->      A
        # |  |  x  x     # |  |  x  x
        # 0, 0, 0, 0     # 0, 0, 1, 1
        # 0, 1, 0, 0     # 0, 1, 1, 1
        # 1, 0, 0, 0     # 1, 0, 1, 1
        # 1, 1, 0, 0     # 1, 1, 1, 1

        example is for n=4, next_min_active=2 (x for A), prev_max_active=2 (| for B)
        '|' indicates that these can take on any given values while 'x' means that is static.

        """
        next = create_next_active_vectors(n, next_min_active)
        prev = create_prev_active_end(n, prev_max_active)
        a_next = combine_speakers(next, prev)

        # symmetric
        b_next = torch.stack((a_next[:, 1], a_next[:, 0]), dim=1)
        return torch.stack((a_next, b_next))

    def get_single_active(n=4):
        v = get_all_permutations(n, 1)
        a_next = combine_speakers(
            v,
            torch.zeros(
                n,
            ),
        )
        b_next = torch.stack((a_next[:, 1], a_next[:, 0]), dim=1)
        return torch.stack((a_next, b_next))

    def get_single_active_idx(vad_projection_head):
        single_speaker = get_single_active(n=len(vad_projection_head.bin_sizes))
        single_speaker = vad_projection_head.onehot_to_idx(single_speaker)

        # sort for convenience
        a_only, _ = single_speaker[0].sort()
        b_only, _ = single_speaker[1].sort()
        return torch.stack((a_only, b_only))

    def get_end_of_segment(n, max=3):
        v = torch.zeros((max + 1, n))
        for i in range(max):
            v[i + 1, : i + 1] = 1
        return v

    #######################################################################
    def test_all_unique(idx):
        assert len(idx) == len(idx.unique()), "contains replicates"

    def find_matches(x, idx):
        return (x.unsqueeze(-1) == idx.flatten(0)).sum(dim=-1)

    def fill_pauses(vad, prev_speaker, next_speaker, ds):
        fill_hold = vad.clone()
        silence = ds == 1
        same_next_prev = prev_speaker == next_speaker
        holds_oh = torch.logical_and(silence, same_next_prev)
        for speaker in [0, 1]:
            fill_oh = torch.logical_and(holds_oh, next_speaker == speaker)
            fill = torch.where(fill_oh)
            fill_hold[(*fill, [speaker] * len(fill[0]))] = 1
        return fill_hold

    def find_valid_silences(vad, horizon=150, min_context_frames=0):
        max_frames = vad.shape[1] - horizon

        # Fill pauses where appropriate
        ###############################################
        prev_speaker = VAD.get_last_speaker(vad)
        next_speaker = VAD.get_next_speaker(vad)
        ds = VAD.vad_to_dialog_vad_states(vad)

        ###############################################
        fill_hold = fill_pauses(vad, prev_speaker, next_speaker, ds)

        ###############################################
        valid = torch.zeros(vad.shape[:-1], device=vad.device)
        for nb in range(ds.shape[0]):
            s, d, v = find_island_idx_len(ds[nb])
            if v[-1] == 1:
                # if segment ends in mutual silence we can't
                # lookahead what happens after
                # thus we omit the last entry
                s = s[:-1]
                d = d[:-1]
                v = v[:-1]
            sil = torch.where(v == 1)[0]
            sil_start = s[sil]
            sil_dur = d[sil]
            after_sil = s[sil + 1]  # this can break
            for ii, start in enumerate(after_sil):
                if start >= max_frames:
                    break
                if start <= min_context_frames:
                    continue
                total_activity_window = fill_hold[nb, start : start + horizon].sum(
                    dim=0
                )
                # a single channel has no activity
                if (total_activity_window == 0).sum() == 1:
                    vs = sil_start[ii]
                    ve = vs + sil_dur[ii]
                    valid[nb, vs:ve] = 1
        return valid

        # disentangle shift/hold
        a_prev = prev_speaker == 0
        a_next = next_speaker == 0
        b_prev = prev_speaker == 1
        b_next = next_speaker == 1

        w_ab = torch.where(torch.logical_and(a_prev, b_next))
        w_aa = torch.where(torch.logical_and(a_prev, a_next))
        w_ba = torch.where(torch.logical_and(b_prev, a_next))
        w_bb = torch.where(torch.logical_and(b_prev, b_next))

        shifts = torch.zeros(vad.shape[:-1], device=vad.device)
        shifts[w_ab] = valid[w_ab]
        shifts[w_ba] = valid[w_ba]

        holds = torch.zeros(vad.shape[:-1], device=vad.device)
        holds[w_aa] = valid[w_aa]
        holds[w_bb] = valid[w_bb]
        return valid, shifts, holds

    def get_locations(vad):
        prev_speaker = VAD.get_last_speaker(vad)
        next_speaker = VAD.get_next_speaker(vad)
        a_next = next_speaker == 0
        a_prev = prev_speaker == 0
        b_next = next_speaker == 1
        b_prev = prev_speaker == 1
        ab = torch.where(torch.logical_and(a_prev, b_next))
        ba = torch.where(torch.logical_and(b_prev, a_next))
        aa = torch.where(torch.logical_and(a_prev, a_next))
        bb = torch.where(torch.logical_and(b_prev, b_next))
        return {"ab": ab, "ba": ba, "aa": aa, "bb": bb}

    #####################################################################
    def plot_area(oh, ax, label=None, color="b", alpha=1):
        ax.fill_between(
            torch.arange(oh.shape[0]),
            y1=-1,
            y2=1,
            where=oh,
            color=color,
            alpha=alpha,
            label=label,
        )

    def extract_hold_shift_probs(probs, vad, valid, all_shift_hold, shift_scale=1.0):
        """hold/shift probs"""
        # extract next/previous speaker information
        prev_speaker = VAD.get_last_speaker(vad)
        next_speaker = VAD.get_next_speaker(vad)
        a_next = next_speaker == 0
        a_prev = prev_speaker == 0
        b_next = next_speaker == 1
        b_prev = prev_speaker == 1
        ab = torch.logical_and(a_prev, b_next)
        ba = torch.logical_and(b_prev, a_next)
        aa = torch.logical_and(a_prev, a_next)
        bb = torch.logical_and(b_prev, b_next)

        # Combine all valid silences with last/previous speaker
        # to get Shift/Holds and associated predictions
        w_ab = torch.where(torch.logical_and(valid, ab))
        w_ba = torch.where(torch.logical_and(valid, ba))
        w_aa = torch.where(torch.logical_and(valid, aa))
        w_bb = torch.where(torch.logical_and(valid, bb))

        ####################################
        # c_ab = change_speaker[0] * shift_scale
        # c_ba = change_speaker[1] * shift_scale

        # Renormalize probabilites
        split = len(all_shift_hold) // 2

        pp = probs[..., all_shift_hold]
        p_a = pp[..., :split].sum(dim=-1) * shift_scale
        p_b = pp[..., split:].sum(dim=-1)
        p_sum = p_a + p_b
        p_a /= p_sum
        p_b /= p_sum
        # pp /= pp.sum(dim=-1, keepdim=True)  # renormalize
        # probs_a_next = pp[..., :split].sum(dim=-1)
        # probs_b_next = pp[..., split:].sum(dim=-1)

        shifts = torch.zeros_like(p_a)
        holds = torch.zeros_like(p_a)

        shift_probs = torch.zeros_like(p_a)
        hold_probs = torch.zeros_like(p_a)

        # Shifts
        if len(w_ab[0]) > 0:
            # sp = probs_b_next[w_ab]
            sp = p_b[w_ab]
            shift_probs[w_ab] = sp
            hold_probs[w_ab] = 1 - sp
            shifts[w_ab] = 1

        if len(w_ba[0]) > 0:
            # sp = probs_a_next[w_ba]
            sp = p_a[w_ba]
            shift_probs[w_ba] = sp
            hold_probs[w_ba] = 1 - sp
            shifts[w_ba] = 1

        # Holds
        if len(w_aa[0]) > 0:
            hp = p_a[w_aa]
            # hp = probs_a_next[w_aa]
            hold_probs[w_aa] = hp
            shift_probs[w_aa] = 1 - hp
            holds[w_aa] = 1

        if len(w_bb[0]) > 0:
            hp = p_b[w_bb]
            # hp = probs_b_next[w_bb]
            hold_probs[w_bb] = hp
            shift_probs[w_bb] = 1 - hp
            holds[w_bb] = 1

        return holds, shifts, hold_probs, shift_probs

    def extract_change_probs(probs, vad, shift_scale=1.0):
        ds = VAD.vad_to_dialog_vad_states(vad)
        a_current = ds == 0
        b_current = ds == 3
        c_ab = change_speaker[0]
        c_ba = change_speaker[1]
        dc_ab = dont_change_speaker[0]
        dc_ba = dont_change_speaker[1]
        change_probs = torch.zeros(probs.shape[:-1], device=probs.device)

        print(dc_ba)

        # Renormalize
        p_ab = probs[..., c_ab].sum(dim=-1) * shift_scale
        p_aa = probs[..., dc_ab].sum(dim=-1)
        p_sum = p_ab + p_aa
        p_ab /= p_sum

        p_ba = probs[..., c_ba].sum(dim=-1) * shift_scale
        p_bb = probs[..., dc_ba].sum(dim=-1)
        p_sum = p_ba + p_bb
        p_ba /= p_sum

        # CUrrent speaker A
        w_a = torch.where(a_current)
        if len(w_a[0]) > 0:
            # p_ab = probs[w_a][..., c_ab].sum(dim=-1)
            # p_aa = probs[w_a][..., dc_ab].sum(dim=-1)
            # p_tot = p_ab + p_aa
            # p_ab /= p_tot
            # p_aa /= p_tot
            change_probs[w_a] = p_ab[w_a]

        # CUrrent speaker B
        w_b = torch.where(b_current)
        if len(w_b[0]) > 0:
            change_probs[w_b] = p_ba[w_b]
            # p_ba = probs[w_b][..., c_ba].sum(dim=-1)
            # p_bb = probs[w_b][..., dc_ba].sum(dim=-1)
            # p_tot = p_ba + p_bb
            # p_ba /= p_tot
            # p_bb /= p_tot
            # change_probs[w_b]  = p_ba
        return change_probs

    def plot_labels(label_oh, n_rows, n_cols, plot=True):
        j = 0
        fig, ax = plt.subplots(
            n_rows, n_cols, sharex=True, figsize=(4 * n_cols, 2 * n_rows)
        )
        for row in range(n_rows):
            for col in range(n_cols):
                plot_vad_label(label_oh[j], ax=ax[row, col])
                j += 1
                if j >= label_oh.shape[0]:
                    break
            if j >= label_oh.shape[0]:
                break
        plt.tight_layout()
        if plot:
            plt.pause(0.1)
        return fig, ax

    vad_projection_head = VadProjection(bin_sizes)

    plt.close("all")

    # HOLD/SHIFT LABELS
    single_speaker = get_single_active_idx(vad_projection_head)
    single_speaker_oh = vad_projection_head.idx_to_onehot(single_speaker)
    all_single_speaker = single_speaker.flatten()
    print("single_speaker_oh: ", tuple(single_speaker_oh.shape))
    # _ = plot_labels(single_speaker_oh[0], n_rows=4, n_cols=4, plot=True)
    shift_hold_label_oh = get_turn_shift(n, next_min_active=1, prev_max_active=2)
    shift_hold_label = vad_projection_head.onehot_to_idx(shift_hold_label_oh)
    all_shift_hold = shift_hold_label.flatten()
    print(shift_hold_label_oh.shape)
    # _ = plot_labels(shift_hold_label_oh[0], n_rows=4, n_cols=4, plot=True)

    n = 4
    eos = get_end_of_segment(n, max=2)
    nav = create_next_active_vectors(n, min_active=2)
    change_speaker_oh = combine_speakers(eos, nav, mirror=True)
    change_speaker = vad_projection_head.onehot_to_idx(change_speaker_oh)
    print("change_speaker_oh: ", tuple(change_speaker_oh.shape))
    # fig, ax = plot_labels(change_speaker_oh[0], n_rows=3, n_cols=4, plot=True)

    # eos = get_end_of_segment(n, max=4)
    eos = create_next_active_vectors(n, min_active=2)
    # eos = torch.cat((eos, eos_gap))
    zero = torch.zeros((1, n))
    dont_change_speaker_oh = combine_speakers(eos, zero, mirror=True)
    dont_change_speaker = vad_projection_head.onehot_to_idx(dont_change_speaker_oh)
    print("dont_change_speaker_oh: ", tuple(dont_change_speaker_oh.shape))
    # fig, ax = plot_labels(dont_change_speaker_oh[0], n_rows=2, n_cols=2, plot=True)

    # when a is active

    # valid
    forward_horizon = 100
    min_context_frames = 100
    shift_scale = 1

    diter = iter(dm.train_dataloader())

    plt.close("all")

    batch = next(diter)
    with torch.no_grad():
        batch = to_device(batch, model.device)
        loss, out, batch, _ = model.shared_step(batch, reduction="none")
        print("done")
        probs = out["logits_vp"].softmax(dim=-1)
    ###############################################################################
    # plot batch
    vad = batch["vad"]
    valid = find_valid_silences(
        vad, horizon=forward_horizon, min_context_frames=min_context_frames
    )
    holds, shifts, hold_probs, shift_probs = extract_hold_shift_probs(
        probs, vad, valid, all_shift_hold, shift_scale=shift_scale
    )
    change_probs = extract_change_probs(probs, vad, shift_scale=shift_scale)
    ############################################
    b = 0
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    n_fig = 0
    _ = plot_vad_oh(vad[b].cpu(), ax[n_fig], legend_loc="upper right", yticks=[])
    plot_area(shifts[b].cpu(), ax[n_fig], label="Shift", alpha=0.2, color="g")
    plot_area(holds[b].cpu(), ax[n_fig], label="Hold", alpha=0.2, color="r")
    ax[n_fig].legend(loc="upper left")
    n_fig += 1
    ############################################
    _ = plot_vad_oh(
        vad[b].cpu(), ax[n_fig], legend_loc="upper right", yticks=[], alpha=0.1
    )
    twin = ax[n_fig].twinx()
    twin.plot(
        change_probs[b].cpu(),
        label="change %",
        color="g",
        linewidth=2,
        linestyle="dotted",
    )
    twin.plot(shift_probs[b].cpu(), label="shift %", color="g", linewidth=2)
    twin.legend(loc="upper left")
    twin.set_ylim([0, 1])
    n_fig += 1
    plt.pause(0.1)

    # B active
    # B -> A
    # ---------------
    # A:
    # |  |  x  x
    # 0, 0, 1, 1
    # 0, 1, 1, 1
    # 1, 0, 1, 1
    # 1, 1, 1, 1
    # ---------------
    # B:
    # |  |  x  x
    # 0, 0, 0, 0
    # 0, 1, 0, 0
    # 1, 0, 0, 0
    # 1, 1, 0, 0


if __name__ == "__main__":

    from datasets_turntaking import DialogAudioDM
    from conv_ssl.ulm_projection import ULMProjection
    import matplotlib.pyplot as plt

    import matplotlib as mpl

    mpl.use("tkagg")

    chpt = "checkpoints/wav2vec_epoch=6-val_loss=2.42136.ckpt"
    # chpt = "checkpoints/hubert_epoch=18-val_loss=1.61074.ckpt"
    model = ULMProjection.load_from_checkpoint(chpt)
    model.eval
    model.to("cuda")
    data_conf = DialogAudioDM.load_config()
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hz=model.encoder.frame_hz,
        vad_bin_times=data_conf["dataset"]["vad_bin_times"],
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=1,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup()
    diter = iter(dm.val_dataloader())
    bin_sizes = [20, 40, 60, 80]
    if model.encoder.frame_hz == 50:
        bin_sizes = [10, 20, 30, 40]

    batch = next(diter)
    with torch.no_grad():
        batch = to_device(batch, model.device)
        loss, out, batch, _ = model.shared_step(batch, reduction="none")
        print("done")
        probs = out["logits_vp"].softmax(dim=-1)

    vad_projection_head = VadProjection(bin_sizes)

    data, score = vad_projection_head(out["logits_vp"], vad=batch["vad"])
    plt.close("all")
    vad = batch["vad"]
    vad_history = batch["vad_history"]
    vad_label_idx = batch["vad_label"]
    eot_cutoff = 0.5
    # batch score
    acc_shift = score["topk"]["shift"]["topk"] / score["topk"]["shift"]["n"]
    acc_hold = score["topk"]["hold"]["topk"] / score["topk"]["hold"]["n"]
    # EOT
    ds = VAD.vad_to_dialog_vad_states(vad)
    eot_pos_oh = eot_moments(vad, n_frames=100)
    eot_pos_oh = torch.logical_or(eot_pos_oh[0], eot_pos_oh[1])
    eot_pos_oh[ds == 2] = 0
    w_eot_pos = torch.where(eot_pos_oh)
    eot_neg_oh = torch.logical_or(ds == 0, ds == 3)
    eot_neg_oh[w_eot_pos] = 0  # remove positives
    eot_neg_oh[:, -100:] = 0  # unknown future
    eot_probs = data["probs"]["eot"].cpu()
    tp = torch.logical_and(eot_probs >= eot_cutoff, eot_pos_oh).sum()
    fn = torch.logical_and(eot_probs < eot_cutoff, eot_pos_oh).sum()
    tn = torch.logical_and(eot_probs < eot_cutoff, torch.logical_not(eot_pos_oh)).sum()
    fp = torch.logical_and(eot_probs >= eot_cutoff, torch.logical_not(eot_pos_oh)).sum()
    total = tp + fn + tn + fp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print("Acc")
    print("shift: ", acc_shift)
    print("hold: ", acc_hold)
    print("EOT")
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1: ", f1)
    # pred\label | neg  | pos |
    # ------------------------|
    # neg:       | tn   | pos |
    # pos:       | tn   | pos |
    conf = torch.tensor([[tn, fp], [fn, tp]])
    acc_eot = (tn + tp) / total
    pred_pos = eot_probs >= eot_cutoff
    # aggregate
    hold_probs = data["probs"]["hold"]
    shift_probs = data["probs"]["shift"]
    change_probs = data["probs"]["change"]
    eot_probs = data["probs"]["eot"]
    hold_oh, shift_oh = VAD.get_hold_shift_onehot(vad)
    ds = VAD.vad_to_dialog_vad_states(vad)
    b = 0
    fig, ax = plot_vad_projection(
        vad=vad[b].cpu(),
        vad_history=vad_history[b].cpu(),
        shift_probs=shift_probs[b].cpu(),
        hold_probs=hold_probs[b].cpu(),
        change_probs=change_probs[b].cpu(),
        eot_probs=eot_probs[b].cpu(),
        probs=probs[b].cpu(),
        hold_oh=hold_oh[b].cpu(),
        shift_oh=shift_oh[b].cpu(),
        eot_pos_oh=eot_pos_oh[b].cpu(),
        eot_neg_oh=eot_neg_oh[b].cpu(),
        eot_cutoff=eot_cutoff,
        plot=False,
    )
    plt.show()
