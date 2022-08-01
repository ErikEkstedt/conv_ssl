from argparse import ArgumentParser
import parselmouth
import numpy as np
import torchaudio
from os.path import join, basename
from pathlib import Path
from parselmouth.praat import call
import string
from tqdm import tqdm

from conv_ssl.augmentations import torch_to_praat_sound, praat_to_torch
from conv_ssl.evaluation.duration import read_text_grid
from conv_ssl.evaluation.phrase_dataset import PhraseDataset, words_to_vad


ROOT = "assets/phrases_beta"
WAV_ROOT = join(ROOT, "duration_audio")
TG_ROOT = join(ROOT, "duration_alignment")


"""
1. Extract audio and save transcripts -> `python conv_ssl/evaluation/phrases_duration_process.py --preprocess`
2. Align audio and save .TextGrid -> `bash conv_ssl/evaluation/forced_alignment_duration.bash`
    - don't forget to source conda env
"""


def calculate_average_phone_duration(dset):
    phone_durations = {}
    for sample in dset:
        for s, e, p in sample["phones"]:
            d = e - s
            if p in phone_durations:
                phone_durations[p].append(d)
            else:
                phone_durations[p] = [d]
    for phone, durations in phone_durations.items():
        phone_durations[phone] = np.mean(durations)
    return phone_durations


class DurationAvg(object):
    """
    This is not a transformation but a class to change the duration
    on the `phrase dataset`.

    Requires timing of phones
    """

    def __init__(self, phone_durations, sample_rate=16000):
        super().__init__()
        self.phone_durations = phone_durations
        self.sample_rate = sample_rate

        # praat
        self.hop_size = 0.01
        self.f0_min = 60
        self.f0_max = 400
        self.eps = 1e-5

    def __call__(self, sample):
        sound = torch_to_praat_sound(sample["waveform"], sample_rate=self.sample_rate)
        manipulation = call(
            sound, "To Manipulation", self.hop_size, self.f0_min, self.f0_max
        )

        # add the last chunk to keep duration as is
        dur_tier = call(
            manipulation,
            "Create DurationTier",
            "shorten",
            sound.start_time,
            sound.end_time,
        )

        # praat interpolates between start -> point -> end
        # so we add a non-changing duration point before the first
        # phone
        first_phone_start, _, _ = sample["phones"][0]
        first_phone_start = max(first_phone_start - self.eps, 0)
        call(dur_tier, "Add point", first_phone_start, 1.0)

        for start, end, phone in sample["phones"]:
            dur = end - start
            base = self.phone_durations[phone]
            r = dur / base

            # add boundary parts for current phone
            # where the end is slightly before actual end
            # (next point will start exactly on start)
            call(dur_tier, "Add point", start, r)
            call(dur_tier, "Add point", end - self.eps, r)

        # Add a final duration to not change remaining part
        # of audio signal
        _, end, _ = sample["phones"][-1]
        call(dur_tier, "Add point", end, 1.0)

        call([manipulation, dur_tier], "Replace duration tier")
        sound_dur = call(manipulation, "Get resynthesis (overlap-add)")
        return praat_to_torch(sound_dur)


def extract_new_audio(dset):
    """
    Load sample -> change duration -> save .wav and transcript .txt
    """
    phone_durations = calculate_average_phone_duration(dset)
    duration_modifier = DurationAvg(phone_durations)
    Path(WAV_ROOT).mkdir(parents=True, exist_ok=True)
    for sample in tqdm(dset, desc="Extract avg-duration audio"):
        sample["waveform"] = duration_modifier(sample)
        wavfile = join(WAV_ROOT, sample["name"] + ".wav")
        textfile = join(WAV_ROOT, sample["name"] + ".txt")
        torchaudio.save(wavfile, sample["waveform"], sample_rate=dset.sample_rate)
        text = sample["text"].replace("-", " ")
        text = text.translate(str.maketrans("", "", string.punctuation)).lower()
        with open(textfile, "w") as text_file:
            text_file.write(text)
    print("Extracted avg-duration waveforms to -> ", WAV_ROOT)


def raw_sample_to_sample(sample, dset):
    """rewritten `dset.get_sample_data()`"""
    sample["audio_path"] = join(WAV_ROOT, basename(sample["audio_path"]))
    tg_path = join(TG_ROOT, sample["name"] + ".TextGrid")
    tg = read_text_grid(tg_path)
    vad_list = words_to_vad(tg["words"])
    # Returns: waveform, dataset_name, vad, vad_history
    ret = dset._sample_data(sample, vad_list)
    ret["example"] = sample["example"]
    ret["words"] = tg["words"]
    ret["phones"] = tg["phones"]
    ret["size"] = sample["size"]

    # print("ret: ", list(ret.keys()))
    # for k, v in sample.items():
    #     if k in ["vad", "words", "phones", "waveform"]:
    #         continue
    #     ret[k] = v
    # print("ret: ", list(ret.keys()))
    # input()
    return ret


def _test():
    import time

    dset = PhraseDataset("assets/phrases_beta/phrases.json")
    sample = dset.get_sample("student", "long", "female", 3)
    dur_sample = raw_sample_to_sample(sample, dset)
    print("sample['waveform']: ", tuple(sample["waveform"].shape))
    print("dur_sample['waveform']: ", tuple(dur_sample["waveform"].shape))

    for w1, w2 in zip(sample["words"], dur_sample["words"]):
        d1 = w1[1] - w1[0]
        d2 = w2[1] - w2[0]
        # print(w1, d1)
        # print(w2, d2)
        print(d1 - d2)
        print("-" * 30)

    sd.play(sample["waveform"][0], samplerate=16000)
    time.sleep(2.5)
    sd.play(dur_sample["waveform"][0], samplerate=16000)


if __name__ == "__main__":

    import sounddevice as sd

    parser = ArgumentParser()
    parser.add_argument("--process", action="store_true")
    parser.add_argument(
        "--phrases", type=str, default="assets/phrases_beta/phrases.json"
    )

    args = parser.parse_args()
    dset = PhraseDataset(args.phrases)
