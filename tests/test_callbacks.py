import pytest

import random
from copy import deepcopy
import torch
from conv_ssl.callbacks import SymmetricSpeakersCallback


@pytest.mark.callback
@pytest.mark.parametrize("stereo", [True, False])
def test_symmetric_callback(stereo):
    B = 4
    N = 100
    datasets = ["fisher", "switchboard", "callhome"]
    wav_channels = 2 if stereo else 1

    original_batch = {
        "vad": torch.randint(0, 2, (B, N, 2)),
        "vad_history": torch.rand((B, N, 5)),
        "waveform": torch.randn((B, wav_channels, N)),
        "dataset": [random.choice(datasets) for _ in range(B)],
    }
    batch = deepcopy(original_batch)
    clb = SymmetricSpeakersCallback()
    batch = clb.get_symmetric_batch(batch)
    reversed_batch = clb.get_symmetric_batch(deepcopy(batch))

    for k in original_batch.keys():
        orig = original_batch[k]
        flip = batch[k]
        if isinstance(orig, torch.Tensor):
            are_the_same = torch.all(torch.eq(orig, flip))
            if not stereo and k == "waveform":
                assert are_the_same, f"{k} stereo={stereo} are not the same after flip"
                continue
        else:
            are_the_same = orig == flip
            if k == "dataset":
                assert are_the_same, f"{k} are not the same as original after ''flip'"
                continue
        assert not are_the_same, f"{k} are the same as original after ''flip'"

    for k in original_batch.keys():
        orig = original_batch[k]
        rev = reversed_batch[k]
        if isinstance(orig, torch.Tensor):
            are_the_same = torch.all(torch.eq(orig, rev))
            if not stereo and k == "waveform":
                assert (
                    are_the_same
                ), f"{k} stereo={stereo} are not the same after reverse"
                continue
        else:
            are_the_same = orig == rev
        assert are_the_same, f"{k} not same as original after ''reverse'"
