import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from typing import Tuple, Dict

from datasets_turntaking.features.vad import VAD
from conv_ssl.plot_utils import plot_vad_oh
from conv_ssl.utils import find_island_idx_len


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


def plot_vp_head(
    sil_probs, act_probs, vad, valid, hold, shift, area_alpha=0.3, plot=True
):
    sil_probs = sil_probs.cpu()
    act_probs = act_probs.cpu()
    vad = vad.cpu()
    valid = valid.cpu()
    hold = hold.cpu()
    shift = shift.cpu()
    valid_shift = torch.logical_and(valid.unsqueeze(-1), shift.cpu())
    valid_hold = torch.logical_and(valid.unsqueeze(-1), hold.cpu())

    N = vad.shape[0]
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(16, 9))
    ##############################################################################3
    # Next speaker A
    n_fig = 0
    ax[n_fig].plot(sil_probs[:, 0], label="Silence A next", color="b")
    ax[n_fig].plot(
        act_probs[:, 0],
        label="Active A next",
        color="darkblue",
        linestyle="dotted",
    )
    _ = plot_vad_oh(vad.cpu(), ax=ax[n_fig].twinx(), alpha=0.6)
    n_fig += 1
    ##############################################################################3
    # Next speaker B
    ax[n_fig].plot(sil_probs[:, 1], label="Silence B next", color="orange")
    ax[n_fig].plot(
        act_probs[:, 1],
        label="Active B next",
        color="darkorange",
        linestyle="dotted",
    )
    _ = plot_vad_oh(vad.cpu(), ax=ax[n_fig].twinx(), alpha=0.6)
    n_fig += 1
    ##############################################################################3
    # VALID
    _ = plot_vad_oh(vad.cpu(), ax=ax[n_fig].twinx(), alpha=0.6)
    plot_area(valid, ax=ax[n_fig], label="VALID", color="k", alpha=area_alpha)
    ax[n_fig].vlines(
        x=[min_context_frames, N - horizon],
        ymin=-1,
        ymax=1,
        color="k",
        linestyle="dashed",
        linewidth=3,
    )
    n_fig += 1
    ##############################################################################3
    # VALID Hold/Shift
    _ = plot_vad_oh(vad.cpu(), ax=ax[n_fig].twinx(), alpha=0.6)
    plot_area(
        valid_shift[:, 0], ax=ax[n_fig], label="Shift", color="g", alpha=area_alpha
    )
    plot_area(valid_shift[:, 1], ax=ax[n_fig], color="g", alpha=area_alpha)
    plot_area(valid_hold[:, 0], ax=ax[n_fig], label="Hold", color="r", alpha=area_alpha)
    plot_area(valid_hold[:, 1], ax=ax[n_fig], color="r", alpha=area_alpha)
    ax[n_fig].vlines(
        x=[min_context_frames, N - horizon],
        ymin=-1,
        ymax=1,
        color="k",
        linestyle="dashed",
        linewidth=3,
    )
    n_fig += 1
    for a in ax:
        a.legend(loc="upper left", fontsize=12)
    if plot:
        plt.pause(0.1)
    return fig, ax


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


def plot_labels(label_oh, n_rows, n_cols, figtitle=None, plot=True):
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

    if figtitle is not None:
        fig.suptitle(figtitle, fontsize=15, fontweight="bold")

    plt.tight_layout()
    if plot:
        plt.pause(0.1)
    return fig, ax


def plot_all_labels(vad_projection_head, next_speaker=0):
    on_silent_shift_oh = vad_projection_head.idx_to_onehot(
        vad_projection_head.on_silent_shift
    )
    on_silent_hold_oh = vad_projection_head.idx_to_onehot(
        vad_projection_head.on_silent_hold
    )
    on_active_shift_oh = vad_projection_head.idx_to_onehot(
        vad_projection_head.on_active_shift
    )
    on_active_hold_oh = vad_projection_head.idx_to_onehot(
        vad_projection_head.on_active_hold
    )
    print("on_silent_shift_oh: ", tuple(on_silent_shift_oh.shape))
    print("on_silent_hold_oh: ", tuple(on_silent_hold_oh.shape))
    print("on_active_shift_oh: ", tuple(on_active_shift_oh.shape))
    print("on_active_hold_oh: ", tuple(on_active_hold_oh.shape))
    ssh = plot_labels(
        on_silent_shift_oh[next_speaker], n_rows=2, n_cols=2, figtitle="SILENT SHIFT"
    )
    sho = plot_labels(
        on_silent_hold_oh[next_speaker], n_rows=2, n_cols=2, figtitle="SILENT HOLD"
    )
    ash = plot_labels(
        on_active_shift_oh[next_speaker], n_rows=3, n_cols=4, figtitle="ACTIVE SHIFT"
    )
    aho = plot_labels(
        on_active_hold_oh[next_speaker], n_rows=2, n_cols=2, figtitle="ACTIVE HOLD"
    )


def plot_next_speaker_probs(p_next, vad, shift=None, hold=None, plot=True):
    a_prob = p_next[:, 0].cpu()
    b_prob = p_next[:, 1].cpu()
    v = vad.cpu()
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(9, 6))
    ##############################################################################3
    # Next speaker A
    n_fig = 0
    twin = ax[n_fig].twinx()
    _ = plot_vad_oh(v, ax=twin, alpha=0.6)
    if shift is not None:
        plot_area(shift, ax=twin, label="Shift", color="g", alpha=0.2)
    if hold is not None:
        plot_area(hold, ax=twin, label="Shift", color="r", alpha=0.2)
    twin.legend(loc="upper right")
    ax[n_fig].plot(a_prob, label="A", color="b", linewidth=2)
    ax[n_fig].set_ylim([0, 1])

    n_fig += 1
    ##############################################################################3
    # Next speaker B
    twin = ax[n_fig].twinx()
    _ = plot_vad_oh(v, ax=twin, alpha=0.6)
    if shift is not None:
        plot_area(shift, ax=twin, label="Shift", color="g", alpha=0.2)
    if hold is not None:
        plot_area(hold, ax=twin, label="Shift", color="r", alpha=0.2)
    twin.legend(loc="upper right")
    ax[n_fig].plot(b_prob, label="B", color="orange", linewidth=2)
    ax[n_fig].set_ylim([0, 1])
    n_fig += 1
    ##############################################################################3
    # Next speaker B
    diff = a_prob - b_prob
    ax[n_fig].hlines(y=0, xmin=0, xmax=diff.shape[0], color="k", linewidth=2)
    ax[n_fig].fill_between(
        torch.arange(diff.shape[0]),
        y1=0,
        y2=diff,
        where=diff > 0,
        color="b",
    )
    ax[n_fig].fill_between(
        torch.arange(diff.shape[0]),
        y1=diff,
        y2=0,
        where=diff < 0,
        color="orange",
    )
    if shift is not None:
        plot_area(shift, ax=ax[n_fig], label="Shift", color="g", alpha=0.2)
    if hold is not None:
        plot_area(hold, ax=ax[n_fig], label="Shift", color="r", alpha=0.2)
    ax[n_fig].set_ylim([-1, 1])
    if plot:
        plt.pause(0.1)
    return fig, ax


class DialogEvents:
    @staticmethod
    def mutual_silences(vad):
        ds = VAD.vad_to_dialog_vad_states(vad)
        return ds == 1

    @staticmethod
    def single_speaker(vad):
        ds = VAD.vad_to_dialog_vad_states(vad)
        return torch.logical_or(ds == 0, ds == 3)

    @staticmethod
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

    @staticmethod
    def find_valid_silences(
        vad, horizon=150, min_context=0, min_duration=0, start_pad=0, target_frames=-1
    ):
        max_frames = vad.shape[1] - horizon

        # Fill pauses where appropriate
        ###############################################
        prev_speaker = VAD.get_last_speaker(vad)
        next_speaker = VAD.get_next_speaker(vad)
        ds = VAD.vad_to_dialog_vad_states(vad)

        ###############################################
        fill_hold = DialogEvents.fill_pauses(vad, prev_speaker, next_speaker, ds)

        ###############################################
        valid = torch.zeros(vad.shape[:-1], device=vad.device)
        for nb in range(ds.shape[0]):
            s, d, v = find_island_idx_len(ds[nb])
            if min_duration > 0:
                s = s[d >= min_duration]
                v = v[d >= min_duration]
                d = d[d >= min_duration]
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
                if start <= min_context or sil_start[ii] <= min_context:
                    continue
                if start >= max_frames:
                    break
                total_activity_window = fill_hold[nb, start : start + horizon].sum(
                    dim=0
                )
                # a single channel has no activity
                if (total_activity_window == 0).sum() == 1:
                    vs = sil_start[ii]
                    vs += start_pad  # pad to get silence away from last activity
                    end = vs + sil_dur[ii]
                    if target_frames < 0:
                        ve = end
                    else:
                        ve = vs + target_frames
                        if ve > end:
                            continue
                    valid[nb, vs:ve] = 1
        return valid

    @staticmethod
    def find_hold_shifts(vad):
        prev_speaker = VAD.get_last_speaker(vad)
        next_speaker = VAD.get_next_speaker(vad)
        silence = DialogEvents.mutual_silences(vad)

        ab = torch.logical_and(prev_speaker == 0, next_speaker == 1)
        ab = torch.logical_and(ab, silence)
        ba = torch.logical_and(prev_speaker == 1, next_speaker == 0)
        ba = torch.logical_and(ba, silence)
        aa = torch.logical_and(prev_speaker == 0, next_speaker == 0)
        aa = torch.logical_and(aa, silence)
        bb = torch.logical_and(prev_speaker == 1, next_speaker == 1)
        bb = torch.logical_and(bb, silence)

        # we order by NEXT Speaker
        shifts = torch.stack((ba, ab), dim=-1)
        holds = torch.stack((aa, bb), dim=-1)
        return holds, shifts


class ProjectionCodebook(nn.Module):
    def __init__(self, bin_sizes=[20, 40, 60, 80], threshold_ratio=0.5):
        super().__init__()
        self.n_bins = len(bin_sizes) * 2
        self.n_classes = 2 ** self.n_bins
        self.bin_sizes = bin_sizes
        self.horizon = sum(bin_sizes)
        self.threshold_ratio = threshold_ratio

        self.codebook = self.init_codebook()
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

    def forward(self, idx):
        return self.idx_to_onehot(idx)


class VadProjection(ProjectionCodebook):
    def __init__(self, bin_sizes=[20, 40, 60, 80], threshold_ratio=0.5, k=5):
        super().__init__(bin_sizes, threshold_ratio)
        self.k = k

        # indices for extracting turn-taking metrics
        self.on_silent_shift, self.on_silent_hold = self.init_on_silent_shift()
        self.on_active_shift, self.on_active_hold = self.init_on_activity_shift()

    ############# MONO ######################################
    def _all_permutations_mono(self, n, start=0):
        vectors = []
        for i in range(start, 2 ** n):
            i = bin(i).replace("0b", "").zfill(n)
            tmp = torch.zeros(n)
            for j, val in enumerate(i):
                tmp[j] = float(val)
            vectors.append(tmp)
        return torch.stack(vectors)

    def _end_of_segment_mono(self, n, max=3):
        """
        # 0, 0, 0, 0
        # 1, 0, 0, 0
        # 1, 1, 0, 0
        # 1, 1, 1, 0
        """
        v = torch.zeros((max + 1, n))
        for i in range(max):
            v[i + 1, : i + 1] = 1
        return v

    def _on_activity_change_mono(self, n=4, min_active=2):
        """

        Used where a single speaker is active. This vector (single speaker) represents
        the classes we use to infer that the current speaker will end their activity
        and the other take over.

        the `min_active` variable corresponds to the minimum amount of frames that must
        be active AT THE END of the projection window (for the next active speaker).
        This used to not include classes where the activity may correspond to a short backchannel.
        e.g. if only the last bin is active it may be part of just a short backchannel, if we require 2 bins to
        be active we know that the model predicts that the continuation will be at least 2 bins long and thus
        removes the ambiguouty (to some extent) about the prediction.
        """

        base = torch.zeros(n)
        # force activity at the end
        if min_active > 0:
            base[-min_active:] = 1

        # get all permutations for the remaining bins
        permutable = n - min_active
        if permutable > 0:
            perms = self._all_permutations_mono(permutable)
            base = base.repeat(perms.shape[0], 1)
            base[:, :permutable] = perms
        return base

    def _combine_speakers(self, x1, x2, mirror=False):
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

    def _sort_idx(self, x):
        if x.ndim == 1:
            x, _ = x.sort()
        elif x.ndim == 2:
            if x.shape[0] == 2:
                a, _ = x[0].sort()
                b, _ = x[1].sort()
                x = torch.stack((a, b))
            else:
                x, _ = x[0].sort()
                x = x.unsqueeze(0)
        return x

    ############# MONO ######################################
    def init_on_silent_shift(self):
        """
        During mutual silences we wish to infer which speaker the model deems most likely.

        We focus on classes where only a single speaker is active in the projection window,
        renormalize the probabilities on this subset, and determine which speaker is the most
        likely next speaker.
        """

        n = len(self.bin_sizes)

        # active channel: At least 1 bin is active -> all permutations (all except the no-activity)
        # active = self._all_permutations_mono(n, start=1)  # at least 1 active
        # active channel: At least 1 bin is active -> all permutations (all except the no-activity)
        active = self._on_activity_change_mono(n, min_active=2)
        # non-active channel: zeros
        non_active = torch.zeros((1, active.shape[-1]))
        # combine
        shift_oh = self._combine_speakers(active, non_active, mirror=True)
        shift = self.onehot_to_idx(shift_oh)
        shift = self._sort_idx(shift)

        # symmetric, this is strictly unneccessary but done for convenience and to be similar
        # to 'get_on_activity_shift' where we actually have asymmetric classes for hold/shift
        hold = shift.flip(0)
        return shift, hold

    def init_on_activity_shift(self):
        n = len(self.bin_sizes)

        # Shift subset
        eos = self._end_of_segment_mono(n, max=2)
        nav = self._on_activity_change_mono(n, min_active=2)
        shift_oh = self._combine_speakers(nav, eos, mirror=True)
        shift = self.onehot_to_idx(shift_oh)
        shift = self._sort_idx(shift)

        # Don't shift subset
        eos = self._on_activity_change_mono(n, min_active=2)
        zero = torch.zeros((1, n))
        hold_oh = self._combine_speakers(zero, eos, mirror=True)
        hold = self.onehot_to_idx(hold_oh)
        hold = self._sort_idx(hold)
        return shift, hold

    #############################################################
    def get_marginal_probs(self, probs, pos_idx, neg_idx):
        p = []
        for next_speaker in [0, 1]:
            joint = torch.cat((pos_idx[next_speaker], neg_idx[next_speaker]), dim=-1)
            p_sum = probs[..., joint].sum(dim=-1)
            p.append(probs[..., pos_idx[next_speaker]].sum(dim=-1) / p_sum)
        return torch.stack(p, dim=-1)

    def get_silence_shift_probs(self, probs):
        return self.get_marginal_probs(probs, self.on_silent_shift, self.on_silent_hold)

    def get_active_shift_probs(self, probs):
        return self.get_marginal_probs(probs, self.on_active_shift, self.on_active_hold)

    def get_next_speaker_probs(self, probs, vad):
        sil_probs = self.get_silence_shift_probs(probs)
        act_probs = self.get_active_shift_probs(probs)

        p_a = torch.zeros_like(sil_probs[..., 0])
        p_b = torch.zeros_like(sil_probs[..., 0])

        # dialog states
        ds = VAD.vad_to_dialog_vad_states(vad)
        silence = ds == 1
        a_current = ds == 0
        b_current = ds == 3
        both = ds == 2

        # silence
        w = torch.where(silence)
        p_a[w] = sil_probs[w][..., 0]
        p_b[w] = sil_probs[w][..., 1]

        # A current speaker
        w = torch.where(a_current)
        p_b[w] = act_probs[w][..., 1]
        p_a[w] = 1 - act_probs[w][..., 1]

        # B current speaker
        w = torch.where(b_current)
        p_a[w] = act_probs[w][..., 0]
        p_b[w] = 1 - act_probs[w][..., 0]

        # Both
        w = torch.where(both)
        # Re-Normalize and compare next-active
        sum = act_probs[w][..., 0] + act_probs[w][..., 1]
        p_a[w] = act_probs[w][..., 0] / sum
        p_b[w] = act_probs[w][..., 1] / sum
        return torch.stack((p_a, p_b), dim=-1)


class VadProjectionOLD(ProjectionCodebook):
    def __init__(self, bin_sizes=[20, 40, 60, 80], threshold_ratio=0.5, k=5):
        super().__init__(bin_sizes, threshold_ratio)
        self.k = k

        # Onehot-representation vectors
        self.first_speaker_idx = self.init_first_speaker_mapping()
        self.next_speaker_idx = self.init_next_speaker_idx()
        self.eot_idx = self.init_eot_idx()

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


def OLD():
    data, score = vad_projection_head(out["logits_vp"], vad=batch["vad"])
    plt.close("all")
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


def explanation():
    vad_projection_head = VadProjection(bin_sizes)
    print("on_silent_shift: ", tuple(vad_projection_head.on_silent_shift.shape))
    print("on_silent_hold: ", tuple(vad_projection_head.on_silent_hold.shape))
    print("on_active_shift: ", tuple(vad_projection_head.on_active_shift.shape))
    print("on_active_hold: ", tuple(vad_projection_head.on_active_hold.shape))
    print("--------------------------------------------")
    plot_all_labels(vad_projection_head)


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

    horizon = 70  # 1s
    min_context = 100  # 1s
    min_duration = 20  # 200ms
    vad_projection_head = VadProjection(bin_sizes)

    batch = next(diter)
    with torch.no_grad():
        batch = to_device(batch, model.device)
        loss, out, batch, _ = model.shared_step(batch, reduction="none")
        probs = out["logits_vp"].softmax(dim=-1)
        vad = batch["vad"]
        vad_history = batch["vad_history"]
        vad_label = batch["vad_label"]
    # Probs
    p_next = vad_projection_head.get_next_speaker_probs(probs, vad).cpu()
    # TEST PLACES
    # Valid shift/hold
    valid = DialogEvents.find_valid_silences(
        vad,
        horizon,
        min_context=min_context,
        min_duration=min_duration,
        start_pad=10,
        target_frames=10,
    )
    hold, shift = DialogEvents.find_hold_shifts(vad)
    hold, shift = torch.logical_and(hold, valid.unsqueeze(-1)), torch.logical_and(
        shift, valid.unsqueeze(-1)
    )
    b = 0
    _ = plot_next_speaker_probs(
        p_next[b].cpu(),
        vad[b].cpu(),
        shift=shift[b].sum(dim=-1).cpu(),
        hold=hold[b].sum(dim=-1).cpu(),
    )

    silence = DialogEvents.mutual_silences(vad)

    # extract_shift(p_next, where=silence):
    # Metrics get shift/hold
    prev_speaker = VAD.get_last_speaker(vad)
    next_speaker = VAD.get_next_speaker(vad)
    where = valid.clone()

    # collect prev/next speakers as hold/shift
    ab = torch.logical_and(prev_speaker == 0, next_speaker == 1)
    ba = torch.logical_and(prev_speaker == 1, next_speaker == 0)
    aa = torch.logical_and(prev_speaker == 0, next_speaker == 0)
    bb = torch.logical_and(prev_speaker == 1, next_speaker == 1)
    # valid locations
    ab = torch.logical_and(ab, where)
    ba = torch.logical_and(ba, where)
    aa = torch.logical_and(aa, where)
    bb = torch.logical_and(bb, where)

    plt.close("all")

    # Dataset can use 'valid' + hold/shift to only extract short
    # windows at appropriate times. may balance between shift/hold

    # Is RNN better at not be "slow" at turn-shifts?
    # what effect does sequence length have on performance?
    # How much context should be the minimum for turn-taking? 3 sec?

    # metrics
    # * one single frame
    # * aggregate
    # * aggregate + single chunk

    ################################################################################
    # b = 0
    # sil_probs = vad_projection_head.get_silence_shift_probs(probs).cpu()
    # act_probs = vad_projection_head.get_active_shift_probs(probs).cpu()
    # fig, ax = plot_vp_head(
    #     sil_probs[b], act_probs[b], vad[b], valid[b], hold[b], shift[b], plot=True
    # )
