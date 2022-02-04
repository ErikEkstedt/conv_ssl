import torch
import torch.nn as nn
from einops import rearrange
from torchmetrics import Metric

from typing import Union, List

from conv_ssl.utils import find_island_idx_len
from datasets_turntaking.features.vad import VAD


def time_to_frames(time, frame_hz):
    if isinstance(time, list):
        time = torch.tensor(time)

    frame = time * frame_hz

    if isinstance(frame, torch.Tensor):
        frame = frame.long().tolist()
    else:
        frame = int(frame)

    return frame


class DialogEvents:
    @staticmethod
    def mutual_silences(vad, ds=None):
        if ds is None:
            ds = VAD.vad_to_dialog_vad_states(vad)
        return ds == 1

    @staticmethod
    def single_speaker(vad, ds=None):
        if ds is None:
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
        # ds = ds.cpu()

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

            if len(s) < 1:
                continue

            sil = torch.where(v == 1)[0]
            sil_start = s[sil]
            sil_dur = d[sil]
            after_sil = s[sil + 1]
            for ii, start in enumerate(after_sil):
                if start <= min_context:
                    continue
                if sil_start[ii] <= min_context:
                    continue
                if start >= max_frames:
                    break

                total_activity_window = fill_hold[nb, start : start + horizon].sum(
                    dim=0
                )
                # a single channel has no activity
                if (total_activity_window == 0).sum() == 1:
                    if sil_dur[ii] < min_duration:
                        continue

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
        return valid, prev_speaker, next_speaker, ds

    @staticmethod
    def find_hold_shifts(vad, prev_speaker=None, next_speaker=None, silence=None):
        if prev_speaker is None:
            prev_speaker = VAD.get_last_speaker(vad)
        if next_speaker is None:
            next_speaker = VAD.get_next_speaker(vad)
        if silence is None:
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

    @staticmethod
    def on_silence(vad, start_pad, target_frames, horizon, min_context, min_duration):
        valid, prev_speaker, next_speaker, ds = DialogEvents.find_valid_silences(
            vad,
            horizon=horizon,
            min_context=min_context,
            min_duration=min_duration,
            start_pad=start_pad,
            target_frames=target_frames,
        )
        hold, shift = DialogEvents.find_hold_shifts(
            vad, prev_speaker=prev_speaker, next_speaker=next_speaker, silence=ds == 1
        )
        return torch.logical_and(hold, valid.unsqueeze(-1)), torch.logical_and(
            shift, valid.unsqueeze(-1)
        )


class VadLabel:
    def __init__(self, bin_times=[0.2, 0.4, 0.6, 0.8], vad_hz=100, threshold_ratio=0.5):
        self.bin_times = bin_times
        self.vad_hz = vad_hz
        self.threshold_ratio = threshold_ratio

        self.bin_sizes = time_to_frames(bin_times, vad_hz)
        self.n_bins = len(self.bin_sizes)
        self.total_bins = self.n_bins * 2
        self.horizon = sum(self.bin_sizes)

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
        return rearrange(v_bins, "... (c d) -> ... c d", c=2)

    def vad_projection(self, vad) -> torch.Tensor:
        """
        Given a sequence of binary VAD information (two channels) we extract a prediction horizon
        (frame length = the sum of all bin_sizes).

        ! WARNING ! VAD is shifted one step to get the 'next frame horizon'

        ```python
        # vad: (B, N, 2)
        # DO THIS
        vad_projection_oh = VadProjection.vad_to_idx(vad)
        # vad_projection_oh: (B, N, 2, )
        ```

        Arguments:
            vad:        torch.Tensor, (b, n, c) or (n, c)
        """
        # (b, n, c) -> (b, N, c, M), M=horizon window size, N=valid frames
        # Shift to get next frame projections
        vv = vad[..., 1:, :]
        vad_projections = vv.unfold(dimension=-2, size=sum(self.bin_sizes), step=1)

        # (b, N, c, M) -> (B, N, 2, len(self.bin_sizes))
        v_bins = self.horizon_to_onehot(vad_projections)
        return v_bins


class ProjectionCodebook(nn.Module):
    def __init__(self, bin_times=[0.20, 0.40, 0.60, 0.80], frame_hz=100):
        super().__init__()
        self.frame_hz = frame_hz
        self.bin_sizes = time_to_frames(bin_times, frame_hz)

        self.n_bins = len(bin_times)
        self.total_bins = self.n_bins * 2
        self.n_classes = 2 ** self.total_bins

        self.codebook = self.init_codebook()
        self.on_silent_shift, self.on_silent_hold = self.init_on_silent_shift()
        self.on_active_shift, self.on_active_hold = self.init_on_activity_shift()
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
            Create a matrix of all one-hot encodings representing a binary sequence of `self.total_bins` places
            Useful for usage in `nn.Embedding` like module.
            """
            n_codes = 2 ** n_bins
            embs = torch.zeros((n_codes, n_bins))
            for i in range(2 ** n_bins):
                embs[i] = single_idx_to_onehot(i, d=n_bins)
            return embs

        codebook = nn.Embedding(
            num_embeddings=self.n_classes, embedding_dim=self.total_bins
        )
        codebook.weight.data = create_code_vectors(self.total_bins)
        codebook.weight.requires_grad_(False)
        return codebook

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

    def init_on_silent_shift(self):
        """
        During mutual silences we wish to infer which speaker the model deems most likely.

        We focus on classes where only a single speaker is active in the projection window,
        renormalize the probabilities on this subset, and determine which speaker is the most
        likely next speaker.
        """

        # active channel: At least 1 bin is active -> all permutations (all except the no-activity)
        # active = self._all_permutations_mono(n, start=1)  # at least 1 active
        # active channel: At least 1 bin is active -> all permutations (all except the no-activity)
        active = self._on_activity_change_mono(self.n_bins, min_active=2)
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
        # Shift subset
        eos = self._end_of_segment_mono(self.n_bins, max=2)
        nav = self._on_activity_change_mono(self.n_bins, min_active=2)
        shift_oh = self._combine_speakers(nav, eos, mirror=True)
        shift = self.onehot_to_idx(shift_oh)
        shift = self._sort_idx(shift)

        # Don't shift subset
        eos = self._on_activity_change_mono(self.n_bins, min_active=2)
        zero = torch.zeros((1, self.n_bins))
        hold_oh = self._combine_speakers(zero, eos, mirror=True)
        hold = self.onehot_to_idx(hold_oh)
        hold = self._sort_idx(hold)
        return shift, hold

    def onehot_to_idx(self, x) -> torch.Tensor:
        """
        The inverse of the 'forward' function.

        Arguments:
            x:          torch.Tensor (*, 2, 4)

        Inspiration for distance calculation:
            https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/vector_quantize_pytorch.py
        """
        assert x.shape[-2:] == (2, self.n_bins)

        # compare with codebook and get closest idx
        shape = x.shape
        flatten = rearrange(x, "... c bpp -> (...) (c bpp)", c=2, bpp=self.n_bins)
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
        return rearrange(v, "... (c b) -> ... c b", c=2)

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

    def get_next_speaker_probs(self, logits, vad):
        probs = logits.softmax(dim=-1)
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

    def speaker_prob_to_shift(self, probs, vad):
        """
        next speaker probabilities (B, N, 2) -> turn-shift probabilities (B, n)
        """
        assert probs.ndim == 3, "Assumes probs.shape = (B, N, 2)"

        shift_probs = torch.zeros(probs.shape[:-1])

        # dialog states
        ds = VAD.vad_to_dialog_vad_states(vad)
        silence = ds == 1
        a_current = ds == 0
        b_current = ds == 3
        prev_speaker = VAD.get_last_speaker(vad)

        # A active -> B = 1 is next_speaker
        w = torch.where(a_current)
        shift_probs[w] = probs[w][..., 1]
        # B active -> A = 0 is next_speaker
        w = torch.where(b_current)
        shift_probs[w] = probs[w][..., 0]
        # silence and A was previous speaker -> B = 1 is next_speaker
        w = torch.where(torch.logical_and(silence, prev_speaker == 0))
        shift_probs[w] = probs[w][..., 1]
        # silence and B was previous speaker -> A = 0 is next_speaker
        w = torch.where(torch.logical_and(silence, prev_speaker == 1))
        shift_probs[w] = probs[w][..., 0]
        return shift_probs

    def forward(self, projection_window):
        # return self.idx_to_onehot(idx)
        return self.onehot_to_idx(projection_window)


class ShiftHoldMetric(Metric):
    """Used in conjuction with 'VadProjection' from datasets_turntaking"""

    def __init__(
        self,
        horizon=1,
        min_context=1,
        start_pad=0.25,
        target_duration=0.05,
        frame_hz=100,
        dist_sync_on_step=False,
    ):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("hold_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("hold_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("shift_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("shift_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.frame_horizon = time_to_frames(horizon, frame_hz)
        self.frame_min_context = time_to_frames(min_context, frame_hz)
        self.frame_start_pad = time_to_frames(start_pad, frame_hz)
        self.frame_target_duration = time_to_frames(target_duration, frame_hz)
        self.frame_min_duration = self.frame_start_pad + self.frame_target_duration
        self.pred_threshold = 0.5

    def stats(self, ac, an, bc, bn):
        """
        F1 statistics over Shift/Hold

        Example 'Shift':
            * ac = shift_correct
            * an = shift_total
            * bc = hold_correct
            * bn = hold_total

        True Positives:  shift_correct
        False Negatives:  All HOLD predictions at SHIFT locations -> (shift_total - shift_correct)
        True Negatives:  All HOLD predictions at HOLD locations -> hold_correct
        False Positives:  All SHIFT predictions at HOLD locations -> (hold_total - hold_correct)

        Symmetrically true for Holds.
        """
        EPS = 1e-9
        tp = ac
        fn = an - ac
        tn = bc
        fp = bn - bc
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        support = tp + fn
        f1 = tp / (tp + 0.5 * (fp + fn) + EPS)
        return {
            "f1": f1,
            "support": support,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    def compute(self):
        """Compute final result"""
        stats = {
            "hold": self.stats(
                ac=self.hold_correct,
                an=self.hold_total,
                bc=self.shift_correct,
                bn=self.shift_total,
            ),
            "shift": self.stats(
                ac=self.shift_correct,
                an=self.shift_total,
                bc=self.hold_correct,
                bn=self.hold_total,
            ),
        }

        # Weighted F1 score
        # scaled/weighted by the support of each metric
        # shift_f1*shift_support + hold_f1*hold_support )/ (shift_support + hold_support)
        f1h = stats["hold"]["f1"] * stats["hold"]["support"]
        f1s = stats["shift"]["f1"] * stats["shift"]["support"]
        tot = stats["hold"]["support"] + stats["shift"]["support"]
        stats["f1_weighted"] = (f1h + f1s) / tot
        return stats

    def extract_acc(self, p_next, shift, hold):
        ret = {
            "shift": {"correct": 0, "n": 0},
            "hold": {"correct": 0, "n": 0},
        }
        # shifts
        next_speaker = 0
        w = torch.where(shift[..., next_speaker])
        if len(w[0]) > 0:
            sa = (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            ret["shift"]["correct"] += sa
            ret["shift"]["n"] += len(w[0])
        next_speaker = 1
        w = torch.where(shift[..., next_speaker])
        if len(w[0]) > 0:
            sb = (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            ret["shift"]["correct"] += sb
            ret["shift"]["n"] += len(w[0])
        # holds
        next_speaker = 0
        w = torch.where(hold[..., next_speaker])
        if len(w[0]) > 0:
            ha = (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            ret["hold"]["correct"] += ha
            ret["hold"]["n"] += len(w[0])
        next_speaker = 1
        w = torch.where(hold[..., next_speaker])
        if len(w[0]) > 0:
            hb = (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            ret["hold"]["correct"] += hb
            ret["hold"]["n"] += len(w[0])
        return ret

    def stats_update(
        self,
        hold_correct,
        hold_total,
        shift_correct,
        shift_total,
    ):
        self.hold_correct += hold_correct
        self.hold_total += hold_total
        self.shift_correct += shift_correct
        self.shift_total += shift_total

    def update(self, p_next, vad):
        hold, shift = DialogEvents.on_silence(
            vad,
            start_pad=self.frame_start_pad,
            target_frames=self.frame_target_duration,
            horizon=self.frame_horizon,
            min_context=self.frame_min_context,
            min_duration=self.frame_min_duration,
        )

        # extract TP, FP, TN, FN
        m = self.extract_acc(p_next, shift, hold)

        self.stats_update(
            hold_correct=m["hold"]["correct"],
            hold_total=m["hold"]["n"],
            shift_correct=m["shift"]["correct"],
            shift_total=m["shift"]["n"],
        )


##########################################
def time_label_making():
    import time

    vad = torch.randint(0, 2, (32, 1000, 2))

    VL = VadLabel(bin_times=[0.2, 0.4, 0.6, 0.8], vad_hz=FRAME_HZ)

    # Time label making
    t = time.time()
    for i in range(10):
        lab_oh = VL.vad_to_label_oh(vad)
    t = time.time() - t
    print("bin_times: ", len(VL.bin_times), " took: ", round(t, 4), "seconds")

    VL = VadLabel(bin_times=[0.05] * 60, vad_hz=FRAME_HZ)

    # Time label making
    t = time.time()
    for i in range(10):
        lab_oh = VL.vad_to_label_oh(vad)
    t = time.time() - t
    print("bin_times: ", len(VL.bin_times), " took: ", round(t, 4), "seconds")


if __name__ == "__main__":
    from tqdm import tqdm
    from datasets_turntaking import DialogAudioDM
    from conv_ssl.plot_utils import plot_next_speaker_probs

    FRAME_HZ = 100
    data_conf = DialogAudioDM.load_config()
    # data_conf["dataset"]["type"] = "sliding"
    DialogAudioDM.print_dm(data_conf)
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        audio_context_duration=data_conf["dataset"]["audio_context_duration"],
        ipu_min_time=data_conf["dataset"]["ipu_min_time"],
        ipu_pause_time=data_conf["dataset"]["ipu_pause_time"],
        vad_hz=FRAME_HZ,
        vad_bin_times=data_conf["dataset"]["vad_bin_times"],
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=32,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup()

    metric = ShiftHoldMetric(
        horizon=1,
        min_context=1,
        start_pad=0.25,
        target_duration=0.05,
        frame_hz=FRAME_HZ,
    )

    diter = iter(dm.val_dataloader())

    batch = next(diter)

    regression = True
    cuda = True
    bin_times = [0.2, 0.4, 0.6, 0.8]
    if regression:
        bin_times = [0.05] * 60
    VL = VadLabel(bin_times=bin_times, vad_hz=FRAME_HZ)
    if not regression:
        codebook = ProjectionCodebook(bin_times=bin_times, frame_hz=FRAME_HZ)
        if cuda:
            codebook = codebook.to("cuda")

    if regression:
        logits = torch.rand(
            (batch["vad"].shape[0], batch["vad_history"].shape[1], 2, len(bin_times))
        )
        probs = logits.sigmoid()
        next_probs = probs.mean(dim=-1)
        tot = next_probs.sum(dim=-1)

        # renormalize for comparison
        next_probs[..., 0] = next_probs[..., 0] / tot
        next_probs[..., 1] = next_probs[..., 1] / tot

    else:

        logits = torch.rand(
            (batch["vad"].shape[0], batch["vad_history"].shape[1], codebook.n_classes)
        )
        print("logits: ", logits.shape)
        next_probs = codebook.get_next_speaker_probs(
            logits, batch["vad"][:, : logits.shape[1]]
        )
        print("next_probs: ", tuple(next_probs.shape))

    # regression probs

    import time

    t = time.time()
    batch = next(diter)
    if cuda:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to("cuda")
    vad_projection_window = VL.vad_projection(batch["vad"])
    vad = batch["vad"][:, : vad_projection_window.shape[1]]
    # On Silence/Pauses
    hold, shift = DialogEvents.on_silence(
        vad,
        start_pad=frame_start_pad,
        target_frames=frame_target_duration,
        horizon=frame_horizon,
        min_context=frame_min_context,
        min_duration=frame_min_duration,
    )
    t = time.time() - t
    print(f"(B={vad.shape[0]}) time: {round(t, 4)}")
    print("vad: ", tuple(vad.shape))
    print("hold: ", tuple(hold.shape))
    print("shift: ", tuple(shift.shape))
    print("vad_projection_window: ", tuple(vad_projection_window.shape))
    if not regression:
        proj_idx = codebook(vad_projection_window)
        print("proj_idx: ", tuple(proj_idx.shape))

    # PLOT BATCH
    p_next = torch.zeros_like(vad)
    for b in range(vad.shape[0]):
        _ = plot_next_speaker_probs(
            p_next[b],
            vad=vad[b].cpu(),
            shift=shift[b].sum(dim=-1).cpu(),
            hold=hold[b].sum(dim=-1).cpu(),
        )

    # Skantze bins
    bin_times = torch.tensor([0.1] * 60)
    print(sum(bin_times))
    vad_projection = VadProjection(
        bin_times=bin_times,
        vad_threshold=0.5,
        pred_threshold=0.5,
        event_min_context=1.0,
        event_min_duration=0.15,
        event_horizon=1.0,
        event_start_pad=0.05,
        event_target_duration=0.10,
        frame_hz=100,
    )

    # Standard
    # bin_times = [0.2, 0.4, 1., 1.4]
    # bin_times = [0.2, 0.4, 0.6, 0.8, 1.0]
    bin_times = [0.2, 0.4, 0.6, 0.8]
    print(sum(bin_times))
    vad_projection = VadProjection(
        bin_times=bin_times,
        vad_threshold=0.5,
        pred_threshold=0.5,
        event_min_context=1.0,
        event_min_duration=0.15,
        event_horizon=1.0,
        event_start_pad=0.05,
        event_target_duration=0.10,
        frame_hz=100,
    )

    batch = next(diter)

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    # create vad_labels
    vl = vad_projection.vad_to_label_idx(batch["vad"])
    vloh = vad_projection.vad_to_label_oh(batch["vad"])
    print("batch['vad']: ", tuple(batch["vad"].shape))
    print("vloh: ", tuple(vloh.shape))
    print("vl: ", tuple(vl.shape))

    # # Baseline F0 -> Only HOLD
    # nh = 26010
    # ns = 6105
    # nh = 17152
    # ns = 3751
    # nh = 9178
    # ns = 1869
    # metric = ShiftHoldMetric()
    # metric.shift_correct += 0
    # metric.shift_total += ns
    # metric.hold_correct += nh
    # metric.hold_total += nh
    # r = metric.compute()
    # r
