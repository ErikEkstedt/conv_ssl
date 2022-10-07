import torch
import torchaudio

from os.path import join, basename
from pathlib import Path
from tqdm import tqdm
from parselmouth.praat import call
import parselmouth
import matplotlib.pyplot as plt

try:
    from textgrids import TextGrid
except ModuleNotFoundError:

    def TextGrid(*args, **kwargs):
        assert NotImplementedError(
            "praat-textgrids NOT installed. 'pip install praat-textgrids'"
        )

    pass


from conv_ssl.augmentations import torch_to_praat_sound, praat_to_torch
from conv_ssl.utils import read_txt

"""
* https://github.com/Legisign/Praat-textgrids
    - clone and install `pip install -e .`
    - textgrid
* https://github.com/prosegrinder/python-cmudict
    - `pip install cmudict`
    - syllables
"""


EXAMPLE_TO_TARGET_WORD = {
    "student": "student",
    "psychology": "psychology",
    "first_year": "student",
    "basketball": "basketball",
    "experiment": "before",
    "live": "yourself",
    "work": "side",
    "bike": "bike",
    "drive": "here",
}

# Phones extracted by cmudict
# last syllable decided by me... hope its correct
LAST_SYLLABLE = {
    "student": [["D", "AH0", "N", "T"]],
    "psychology": [["JH", "IY0"]],
    # "year": ["Y", "IH1", "R"]],
    "basketball": [["B", "AO2", "L"]],
    # "experiments": ["M", "AH0", "N", "T", "S"],
    "before": [["B", "IH0", "F", "AO1", "R"], ["B", "IY2", "F", "AO1", "R"]],
    # "live": ["L", "IH1", "V"],
    "yourself": [
        ["Y", "ER0", "S", "EH1", "L", "F"],
        ["Y", "UH0", "R", "S", "EH1", "L", "F"],
        ["Y", "AO1", "R", "S", "EH0", "L", "F"],
    ],
    # "work": ["W", "ER1", "K"],
    "side": [["S", "AY1", "D"]],
    "bike": [["B", "AY1", "K"]],
    # "drive": ["D", "R", "AY1", "V"],
    "here": [["HH", "IY1", "R"]],
}


# TODO: must match with new `read_text_grid`
def match_duration(
    long_waveform, short_phones, long_phones, sample_rate=16000, eps=1e-5, verbose=False
):
    """
    https://www.fon.hum.uva.nl/praat/manual/Intro_8_2__Manipulation_of_duration.html
    https://www.fon.hum.uva.nl/praat/manual/DurationTier.html

    One of the types of objects in Praat. A DurationTier object contains a
    number of (time, duration) points, where duration is to be interpreted
    as a relative duration (e.g. the duration of a manipulated sound as
    compared to the duration of the original). For instance, if
    your DurationTier contains two points, one with a duration value of 1.5
    at a time of 0.5 seconds and one with a duration value of 0.6 at a time
    of 1.1 seconds, this is to be interpreted as a relative duration of 1.5
    (i.e. a slowing down) for all original times before 0.5 seconds, a
    relative duration of 0.6 (i.e. a speeding up) for all original times
    after 1.1 seconds, and a linear interpolation between 0.5 and 1.1
    seconds (e.g. a relative duration of 1.2 at 0.7 seconds, and of 0.9 at 0.9 seconds).


    Match the first phoneme duration of "short" in "long".
    Some example get different phonemes/alignment (they are pronounced differently naturally)
    and if this occurs (3 times in the data) then we simply stop at the last matching phoneme.

    """
    change = []
    for ps, pl in zip(short_phones["intervals"], long_phones["intervals"]):
        pps = ps[-1]
        ppl = pl[-1]
        if pps != ppl:
            # if the phonemes no longer match we break
            continue
        long_dur = pl[1] - pl[0]
        short_dur = ps[1] - ps[0]
        ratio = short_dur / long_dur
        if ratio == 1:
            continue
        change.append([pl[0], pl[1], ratio])

    sound = torch_to_praat_sound(long_waveform, sample_rate=sample_rate)
    manipulation = call(sound, "To Manipulation", 0.01, 60, 400)

    # add the last chunk to keep duration as is
    dur_tier = call(
        manipulation,
        "Create DurationTier",
        "shorten",
        sound.start_time,
        sound.end_time,
    )

    # before this point duration should be the same
    try:
        if change[0][0] > 0:
            call(dur_tier, "Add point", change[0][0] - eps, 1.0)
            if verbose:
                print(f'call(dur_tier, "Add point", {change[0][0]-eps}, 1.)')
    except:
        print(change)
        for ps, pl in zip(short_phones["intervals"], long_phones["intervals"]):
            print(ps[-1], pl[-1])
        input()

    for s, e, r in change:
        call(dur_tier, "Add point", s, r)
        call(dur_tier, "Add point", e, r)
        if verbose:
            print(f'call(dur_tier, "Add point", {s}, {r})')
            print(f'call(dur_tier, "Add point", {e}, {r})')

    # After this point duration should be the same
    call(dur_tier, "Add point", change[-1][1] + eps, 1.0)
    if verbose:
        print(f'call(dur_tier, "Add point", {change[-1][1]+eps}, 1.)')

    call([manipulation, dur_tier], "Replace duration tier")
    sound_dur = call(manipulation, "Get resynthesis (overlap-add)")
    y = sound_dur.as_array().astype("float32")
    return torch.from_numpy(y)


def read_text_grid(path):
    grid = TextGrid(path)
    data = {"words": [], "phones": []}
    for word_phones, vals in grid.items():
        for w in vals:
            if w.text == "":
                continue
            # what about words spoken multiple times?
            # if word_phones == 'words':
            #     data[word_phones][w.text] = (w.xmin, w.xmax)
            data[word_phones].append((w.xmin, w.xmax, w.text))
    return data


def get_word_times(target_word, sample):
    ret = []
    for start, end, word in sample["words"]:
        if word == target_word:
            ret.append((start, end, word))
    return ret


def find_phones_in_interval(sample, start, end):
    phones = []
    for s, e, p in sample["phones"]:
        if start <= s < end and start < e <= end:
            phones.append((s, e, p))
    return phones


def get_last_syllable_duration(sample):
    target_word = EXAMPLE_TO_TARGET_WORD[sample["example"]]
    syllable_list = LAST_SYLLABLE[target_word]

    # Extract target word from sample
    # and take only phones from this word
    word_boundaries = get_word_times(target_word, sample)
    wstart, wend, _ = word_boundaries[0]  # assume only one entry
    phones = find_phones_in_interval(sample, wstart, wend)

    start, end = None, None

    iter_again = False  # hacky
    for syllable in syllable_list:
        n_phones = len(syllable)
        last_phones = phones[-n_phones:]
        start = last_phones[0][0]
        end = last_phones[-1][1]

        # Check that phonemes are the same as expected
        for s, s_phone in zip(syllable, last_phones):
            # assert s == s_phone[-1], f"Not the same phones {syllable} != {last_phones}"
            if s != s_phone[-1]:
                iter_again = True
                break
        if not iter_again:
            break

    assert start is not None, f"start: {start}, end:{end} not Found"
    return start, end, end - start


def extract_final_f0_height(
    waveform, start, end, sample_rate=16000, hop_time=0.01, f0_min=60, f0_max=400
):
    sound = torch_to_praat_sound(waveform, sample_rate)
    pitch = sound.to_pitch(
        time_step=hop_time, pitch_floor=f0_min, pitch_ceiling=f0_max
    ).selected_array["frequency"]

    # Frame boundaries
    s_frame = int(start / hop_time)
    e_frame = int(end / hop_time)

    # stats
    mean = pitch[pitch > 0].mean()
    min_f0 = pitch[pitch > 0].min()
    max_f0 = pitch[s_frame : e_frame + 1].max()
    return max_f0 / mean, max_f0, min_f0, mean


def extract_f0_duration_data(dset):
    data = {}
    n_skipped = 0
    for sample in tqdm(dset, desc="collect f0/dur data"):
        example = sample["example"]
        short_long = sample["size"]
        gender = sample["gender"]

        start, end, dur = get_last_syllable_duration(sample)
        r, f0_max, f0_min, f0_mean = extract_final_f0_height(
            sample["waveform"], start, end
        )

        # add to data
        if example not in data:
            data[example] = {}

        if gender not in data[example]:
            data[example][gender] = {}

        if short_long not in data[example][gender]:
            data[example][gender][short_long] = []

        data[example][gender][short_long].append([dur, r])

    print("Skipped: ", n_skipped)
    return data


def save_all_f0_duration_plots(
    data,
    text=True,
    save=True,
    plot=False,
    savepath="assets/PaperB/eval_phrases/figs/f0_dur",
):
    Path(savepath).mkdir(parents=True, exist_ok=True)
    s = 12
    if text:
        s = 2
    for example, v in data.items():
        fig, ax = plt.subplots(1, 1)
        ax.set_title(example)
        for gender, vv in v.items():
            alpha = 0.8 if gender == "female" else 0.4
            for dur, r in vv["short"]:
                ax.scatter(dur, r, s=s, color="g", alpha=alpha)
                if text:
                    ax.text(dur, r, s="S", color="g", fontweight="bold")
            for dur, r in vv["long"]:
                ax.scatter(dur, r, s=s, color="b", alpha=alpha)
                if text:
                    ax.text(dur, r, s="L", color="b", fontweight="bold")
        ax.set_xlabel("Duration")
        ax.set_ylabel("Rel. F0")
        if save:
            fig.savefig(join(savepath, example + ".png"))
    if plot:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    from conv_ssl.evaluation.phrase_dataset import PhraseDataset

    dset = PhraseDataset(
        "assets/phrases_beta/phrases.json",
        vad_hz=50,
        sample_rate=16000,
        vad_horizon=2.0,
    )

    data = extract_f0_duration_data(dset)

    save_all_f0_duration_plots(data, save=True, plot=True)
