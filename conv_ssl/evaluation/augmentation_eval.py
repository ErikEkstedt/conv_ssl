import torch

"""

From: Learning De-identified Representations of Prosody from Raw Audio


The raw audio is first resampled to 16 kHz, then pitch-shifted on a per-example
basis such that the median pitch of the voice segments is the same value for
the whole dataset. To perform this pitch-shifting, we ran an
autocorrelation-based pitch-tracking method (via Praat (Boersma, 2006)),
calculated the median pitch of the voiced segments in a given sample, then
shifted the pitch such that the median is 150 Hz for the sample.

* A shift operation... All median pitch the same across speakers.
    - "This is primarily to address the distributional difference between sexes in fundamental frequency."

"""

if __name__ == "__main__":
    pass
