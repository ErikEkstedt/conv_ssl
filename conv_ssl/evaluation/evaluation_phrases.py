import torch
import torchaudio
import numpy as np
import librosa
from librosa import display

from pathlib import Path
from os.path import join, basename, exists
from os import makedirs
from copy import deepcopy
from tqdm import tqdm

from os.path import join
import sounddevice as sd

import parselmouth
from parselmouth.praat import call

from conv_ssl.augmentations import (
    torch_to_praat_sound,
    flatten_pitch_batch,
    low_pass_filter_resample,
)
from conv_ssl.evaluation.phrase_dataset import PhraseDataset
from conv_ssl.model import VPModel
from conv_ssl.utils import read_txt, everything_deterministic

import matplotlib.pyplot as plt


everything_deterministic()


def load_model_dset(checkpoint, phrase_path="assets/phrases_beta/phrases.json"):
    model = VPModel.load_from_checkpoint(checkpoint)
    model = model.eval()
    _ = model.to("cuda")

    dset = PhraseDataset(
        phrase_path,
        vad_hz=model.frame_hz,
        sample_rate=model.sample_rate,
        vad_horizon=model.VAP.horizon,
        vad_history=model.conf["data"]["vad_history"],
        vad_history_times=model.conf["data"]["vad_history_times"],
    )
    return model, dset


def create_flat_batch(batch):
    flat_waveform = flatten_pitch_batch(batch["waveform"].unsqueeze(1), batch["vad"])
    flat_batch = {"waveform": flat_waveform}
    for k, v in batch.items():
        if k == "waveform":
            continue
        flat_batch[k] = deepcopy(v)
    return flat_batch


def create_low_pass_batch(batch, cutoff_freq, sample_rate):
    flat_waveform = low_pass_filter_resample(
        batch["waveform"].unsqueeze(1), cutoff_freq, sample_rate
    )
    flat_waveform = flat_waveform.mean(1)
    flat_batch = {"waveform": flat_waveform}
    for k, v in batch.items():
        if k == "waveform":
            continue
        flat_batch[k] = deepcopy(v)
    return flat_batch


def get_hold_prob(p, vad, pre_cutoff=0.5, post_cutoff=0.2, frame_hz=100):
    v = vad[0, :, 0]
    last_voiced_frame = torch.where(v == 1)[0][-1]
    pre_cutoff_frames = last_voiced_frame - int(pre_cutoff * frame_hz)
    post_cutoff_frames = last_voiced_frame + int(post_cutoff * frame_hz)
    pre = p[0, :pre_cutoff_frames, 0]
    end = p[0, pre_cutoff_frames + 1 : last_voiced_frame + 1, 0]
    post = p[0, last_voiced_frame + 1 : post_cutoff_frames + 1, 0]
    return pre, end, post


def extract_phrase_stats(model, dset):
    stats = {}
    for augmentation in ["regular", "flat_pitch", "low_pass"]:
        stats[augmentation] = {"pre": [], "end": []}
        for example, v in tqdm(dset.data.items(), desc=augmentation):
            for gender, sample_list in dset.data[example]["long"].items():
                for ii in range(len(sample_list)):
                    sample = dset.get_sample(example, "long", gender, ii)
                    if augmentation == "flat_pitch":
                        sample = create_flat_batch(sample)
                    elif augmentation == "low_pass":
                        sample = create_low_pass_batch(
                            sample, cutoff_freq=300, sample_rate=model.sample_rate
                        )
                    _, _, prob, sample = model.output(sample)
                    pre, end, post = get_hold_prob(
                        prob["p"],
                        sample["vad"],
                        pre_cutoff=0.5,
                        post_cutoff=0.2,
                        frame_hz=model.frame_hz,
                    )
                    stats[augmentation]["pre"].append(pre)
                    stats[augmentation]["end"].append(end)
        stats[augmentation]["pre"] = torch.cat(stats[augmentation]["pre"])
        stats[augmentation]["end"] = torch.cat(stats[augmentation]["end"])
    return stats


def plot_next_speaker(p_ns, ax, color=["b", "orange"], alpha=0.6):
    x = torch.arange(len(p_ns))
    ax.fill_between(
        x,
        y1=0.5,
        y2=p_ns,
        where=p_ns > 0.5,
        alpha=alpha,
        color=color[0],
        label="A turn",
    )
    ax.fill_between(
        x,
        y1=p_ns,
        y2=0.5,
        where=p_ns < 0.5,
        alpha=alpha,
        color=color[1],
        label="B turn",
    )
    return ax


def plot_sample(p_ns, waveform, words=None, starts=None, sample_rate=16000, plot=False):
    snd = torch_to_praat_sound(waveform, sample_rate)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    pitch_values[pitch_values == 0] = np.nan
    xmin, xmax = snd.xs().min(), snd.xs().max()
    melspec = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=waveform.numpy(), sr=sample_rate, n_fft=800, hop_length=160
        ),
        ref=np.max,
    )[0]

    fig, ax = plt.subplots(4, 1, figsize=(6, 5))
    # Waveform
    ax[0].plot(snd.xs(), snd.values.T, alpha=0.4)
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_ylabel("waveform")
    # SPEC
    img = librosa.display.specshow(
        melspec, x_axis="time", y_axis="mel", sr=sample_rate, fmax=8000, ax=ax[1]
    )
    ax[1].set_xticks([])
    ax[1].set_xlabel("")
    ax[1].set_ylabel("Spec. mel (Hz)")
    ax[1].yaxis.tick_right()
    # Pitch
    ax[2].plot(pitch.xs(), pitch_values, "o", markersize=3, color="b")
    ax[2].set_xlim([xmin, xmax])
    ax[2].set_xticks([])
    ymin, ymax = ax[2].get_ylim()
    if ymax - ymin < 10:
        ymin -= 5
        ymax += 5
        ax[2].set_ylim([ymin, ymax])
    ax[2].yaxis.tick_right()
    ax[2].set_ylabel("Pitch (Hz)")
    # Next speaker probs
    plot_next_speaker(p_ns, ax=ax[3])
    ax[3].set_xlim([0, len(p_ns)])
    ax[3].set_xticks([])
    ax[3].set_yticks([0.25, 0.75], ["SHIFT", "HOLD"])
    ax[3].set_ylim([0, 1])
    ax[3].hlines(y=0.5, xmin=0, xmax=len(p_ns), linestyle="dashed", color="k")

    if starts is not None and words is not None:
        for word, start in zip(words, starts):
            ax[0].text(
                x=start,
                y=-0.5,
                s=word,
                fontsize=12,
                # fontweight='bold',
                rotation=30,
                horizontalalignment="left",
            )
    plt.subplots_adjust(
        left=0.1, bottom=0.05, right=0.9, top=0.95, wspace=None, hspace=0.09
    )

    if plot:
        plt.pause(0.1)
    return fig, ax


def plot_all_samples(model, dset, savepath, checkpoint_name):
    fig_root = join(savepath, checkpoint_name)
    makedirs(savepath, exist_ok=True)
    makedirs(fig_root, exist_ok=True)
    print("root: ", fig_root)
    pbar = tqdm(range(len(dset)))
    for example, v in dset.data.items():
        for short_long, vv in v.items():
            for gender, sample_list in vv.items():
                for nidx in range(len(sample_list)):
                    spath = join(fig_root, example, short_long, gender)
                    if not exists(spath):
                        Path(spath).mkdir(parents=True)
                    sample = dset.get_sample(example, short_long, gender, nidx)
                    flat_sample = create_flat_batch(sample)
                    _, _, prob, sample = model.output(sample)
                    _, _, flat_prob, flat_sample = model.output(flat_sample)
                    fig, _ = plot_sample(
                        prob["p"][0, :, 0],
                        sample["waveform"],
                        words=sample["words"][0],
                        starts=sample["starts"][0],
                    )
                    fig.savefig(join(spath, sample["session"] + ".png"))
                    flat_fig, _ = plot_sample(
                        flat_prob["p"][0, :, 0],
                        flat_sample["waveform"],
                        words=flat_sample["words"][0],
                        starts=flat_sample["starts"][0],
                    )
                    flat_fig.savefig(join(spath, sample["session"] + "_flat.png"))
                    plt.close("all")
                    pbar.update()


##########################################
# DURATION
##########################################


def load_text_grid_phones(path):
    """
    A very hacky way of loading textgrids
    for this particular project... might break (silently) on others...
    """

    def _get_val(line):
        return float(line.split(" ")[-1])

    def _get_string(line):
        return line.split(" ")[-1]

    phone_list = []
    include = False
    for line in read_txt(path):
        if line.startswith("name") and line.endswith('"phones"'):
            include = True
        if include:
            phone_list.append(line)

    phones = {
        "xmin": _get_val(phone_list[1]),
        "xmax": _get_val(phone_list[2]),
        "intervals": [],
    }
    cur_val = []
    for line in phone_list[4:]:
        if line.startswith("intervals") and len(cur_val) > 0:
            phones["intervals"].append(cur_val)
            cur_val = []
            continue
        if line.startswith("xmin"):
            cur_val.append(_get_val(line))
        elif line.startswith("xmax"):
            cur_val.append(_get_val(line))
        elif line.startswith("text"):
            s = _get_string(line).replace('"', "")
            if s == "":
                cur_val = []
            else:
                cur_val.append(s)

    return phones


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


def save_duration_audio(dset, align_root="assets/phrases_beta/alignment"):
    def audio_path_to_align_path(audio_path):
        name = basename(audio_path).replace(".wav", "") + ".TextGrid"
        return join(align_root, name)

    # TODO: pretty much done
    sample_rate = 16000
    for example, v in tqdm(dset.data.items()):
        for gender in ["female", "male"]:
            for idx in range(len(dset.data[example]["short"][gender])):
                s = dset.get_sample(example, "short", gender, idx)
                l = dset.get_sample(example, "long", gender, idx)
                a = s["session"].split("-")[-1]
                b = l["session"].split("-")[-1]
                assert a == b, f"{a}!={b}"
                short_phones = load_text_grid_phones(
                    audio_path_to_align_path(s["audio_path"])
                )
                long_phones = load_text_grid_phones(
                    audio_path_to_align_path(l["audio_path"])
                )
                y = match_duration(
                    l["waveform"], short_phones, long_phones, sample_rate=sample_rate
                )
                if y is None:
                    print(example, gender, idx)
                    input()
                wpath = l["audio_path"].replace(".wav", "_dur_match.wav")
                torchaudio.save(wpath, y, sample_rate=16000)


if __name__ == "__main__":

    ch_root = "assets/PaperB/checkpoints"
    # model = VPModel.load_from_checkpoint(
    #     "assets/PaperB/checkpoints/unused/swb_fis_18ep.ckpt"
    # )
    checkpoint = join(ch_root, "cpc_48_50hz_15gqq5s5.ckpt")
    # checkpoint = "assets/PaperB/checkpoints/cpc_48_20hz_2ucueis8.ckpt"
    # checkpoint = "assets/PaperB/checkpoints/cpc_44_100hz_unfreeze_12ep.ckpt"
    model, dset = load_model_dset(checkpoint)
    checkpoint_name = basename(checkpoint)
    plot_all_samples(model, dset, "assets/PaperB/figs/phrases", checkpoint_name)

    # # sample = dset[6]
    # flat = False
    # if flat:
    #     sample = create_flat_batch(sample)
    # loss, out, prob, sample = model.output(sample)
    # waveform = sample["waveform"]
    # p_ns = prob["p"][0, :, 0]
    # w = sample["words"][0]
    # s = sample["starts"][0]
    # fig, ax = plot_sample(p_ns, waveform, words=w, starts=s)
    # plt.show()

    # all_stats = {}
    # for ckpt in [
    #     "cpc_48_20hz_2ucueis8.ckpt",
    #     "cpc_48_50hz_15gqq5s5.ckpt",
    #     "cpc_44_100hz_unfreeze_12ep.ckpt",
    # ]:
    #     checkpoint = join(ch_root, ckpt)
    #     model, dset = load_model_dset(checkpoint)
    #     checkpoint_name = basename(checkpoint)
    #     stats = extract_phrase_stats(model, dset)
    #     all_stats[checkpoint_name] = {
    #             'pre_mean': stats[]
    #             }
    #     print(checkpoint_name)
    #     print("PRE")
    #     print(
    #         "\tRegular: ", stats["regular"]["pre"].mean(), stats["regular"]["pre"].std()
    #     )
    #     print(
    #         "\tFlat: ",
    #         stats["flat_pitch"]["pre"].mean(),
    #         stats["flat_pitch"]["pre"].std(),
    #     )
    #     print("\tLP: ", stats["low_pass"]["pre"].mean(), stats["low_pass"]["pre"].std())
    #     print("END")
    #     print(
    #         "\tRegular: ", stats["regular"]["end"].mean(), stats["regular"]["end"].std()
    #     )
    #     print(
    #         "\tFlat: ",
    #         stats["flat_pitch"]["end"].mean(),
    #         stats["flat_pitch"]["end"].std(),
    #     )
    #     print("\tLP: ", stats["low_pass"]["end"].mean(), stats["low_pass"]["end"].std())

    # PRE
    #         Regular:  tensor(0.8765) tensor(0.1250)
    #         Flat:  tensor(0.7554) tensor(0.1944)
    #         LP:  tensor(0.8529) tensor(0.0695)
    # END
    #         Regular:  tensor(0.5520) tensor(0.2388)
    #         Flat:  tensor(0.5420) tensor(0.2475)
    #         LP:  tensor(0.8376) tensor(0.0594)
    # cpc_48_50hz_15gqq5s5.ckpt
    # PRE
    #         Regular:  tensor(0.8508) tensor(0.1278)
    #         Flat:  tensor(0.7365) tensor(0.1693)
    #         LP:  tensor(0.9199) tensor(0.0433)
    # END
    #         Regular:  tensor(0.3048) tensor(0.2440)
    #         Flat:  tensor(0.3281) tensor(0.2313)
    #         LP:  tensor(0.9046) tensor(0.0446)
    # cpc_44_100hz_unfreeze_12ep.ckpt
    # PRE
    #         Regular:  tensor(0.8748) tensor(0.0973)
    #         Flat:  tensor(0.8571) tensor(0.1128)
    #         LP:  tensor(0.8670) tensor(0.0721)
    # END
    #         Regular:  tensor(0.4711) tensor(0.2666)
    #         Flat:  tensor(0.6498) tensor(0.2374)
    #         LP:  tensor(0.8321) tensor(0.0758)
