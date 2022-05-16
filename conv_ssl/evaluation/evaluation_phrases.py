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

from conv_ssl.utils import write_json
import conv_ssl.transforms as CT
from conv_ssl.augmentations import torch_to_praat_sound

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


def create_transform_batch(batch, transform):
    n_frames = batch["vad_history"].shape[1]
    vad = batch["vad"][:, :n_frames]
    new_waveform = transform(batch["waveform"].unsqueeze(0), vad=vad)
    new_batch = {"waveform": new_waveform}
    for k, v in batch.items():
        if k == "waveform":
            continue
        new_batch[k] = deepcopy(v)
    return new_batch


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
        y_min = -0.8
        y_max = 0.8
        diff = y_max - y_min
        steps = 4
        for ii, (word, start) in enumerate(zip(words, starts)):
            yy = y_min + diff * (ii % steps) / steps
            ax[0].text(
                x=start + 0.005,
                y=yy,
                s=word,
                fontsize=12,
                horizontalalignment="left",
            )
            ax[0].vlines(
                start, ymin=-1, ymax=1, linestyle="dashed", linewidth=1, color="k"
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


def audio_path_to_align_path(audio_path, root):
    name = basename(audio_path).replace(".wav", "") + ".TextGrid"
    return join(root, name)


def save_duration_audio(dset, align_root="assets/phrases_beta/alignment"):
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


def identity(waveform, vad):
    return waveform.squeeze(0)


def evaluation(model, dset, checkpoint_name, savepath="assets/PaperB/eval_phrases"):
    """
    Aggregate evaluation

    * save global stats
    * save all figures
    """

    transforms = {
        "flat_f0": CT.FlatPitch(),
        "only_f0": CT.LowPass(),
        "shift_f0": CT.ShiftPitch(),
        "flat_intensity": CT.FlatIntensity(vad_hz=model.frame_hz),
        "regular": identity,
    }

    savepath = "assets/PaperB/eval_phrases"
    root = join(savepath, checkpoint_name)
    fig_root = join(root, "figs")
    wav_root = join(root, "audio")
    Path(fig_root).mkdir(parents=True, exist_ok=True)
    Path(wav_root).mkdir(parents=True, exist_ok=True)
    print("root: ", root)
    print("fig root: ", fig_root)
    print("wav root: ", wav_root)

    stats = {"short": {}, "long": {}}
    pbar = tqdm(range(len(dset)))
    for example, v in dset.data.items():
        for short_long, vv in v.items():
            for gender, sample_list in vv.items():
                for nidx in range(len(sample_list)):
                    fig_dir = join(fig_root, example, short_long, gender)
                    wav_dir = join(wav_root, example, short_long, gender)
                    Path(fig_dir).mkdir(parents=True, exist_ok=True)
                    Path(wav_dir).mkdir(parents=True, exist_ok=True)
                    sample = dset.get_sample(example, short_long, gender, nidx)
                    orig_sample = deepcopy(sample)
                    orig_waveform = deepcopy(sample["waveform"].unsqueeze(0))
                    n_frames = sample["vad_history"].shape[1]
                    vad = sample["vad"][:, :n_frames]
                    # name of specific sample file
                    name = f"{example}_{gender}_{short_long}_{sample['tts']}"
                    # save waveform
                    for augmentation, transform in transforms.items():
                        orig_sample["waveform"] = transform(orig_waveform, vad=vad)
                        loss, out, prob, sample = model.output(orig_sample)
                        # Save Figures
                        fig, _ = plot_sample(
                            prob["p"][0, :, 0],
                            sample["waveform"],
                            words=sample["words"][0],
                            starts=sample["starts"][0],
                        )
                        fig.savefig(join(fig_dir, name + f"_{augmentation}.png"))
                        # save waveform
                        wavpath = join(wav_dir, name + f"_{augmentation}.wav")
                        torchaudio.save(
                            wavpath, sample["waveform"], sample_rate=model.sample_rate
                        )
                        # Save Statistics
                        pre, end, post = get_hold_prob(
                            prob["p"],
                            sample["vad"],
                            pre_cutoff=0.5,
                            post_cutoff=0.2,
                            frame_hz=model.frame_hz,
                        )
                        # if short_long == "long":
                        if augmentation not in stats[short_long]:
                            stats[short_long][augmentation] = {"pre": [], "end": []}
                        stats[short_long][augmentation]["pre"].append(pre)
                        stats[short_long][augmentation]["end"].append(end)
                    plt.close("all")
                    pbar.update()

    # Global Stats
    statistics = {}
    for long_short in ["long", "short"]:
        statistics[long_short] = {}
        for augmentation, pre_end in stats[long_short].items():
            pre = 0
            pre_frames = 0
            end = 0
            end_frames = 0
            for s in pre_end["pre"]:
                pre += s.sum()
                pre_frames += len(s)
            for s in pre_end["end"]:
                end += s.sum()
                end_frames += len(s)
            pre /= pre_frames
            end /= end_frames
            pre = pre.item()
            end = 1 - end.item()
            avg = (end + pre) / 2
            statistics[long_short][augmentation] = {"pre": pre, "end": end, "avg": avg}
    torch.save(stats, join(root, "stats.pt"))
    write_json(statistics, join(root, "score.json"))
    print("Saved stats -> ", join(root, "score.json"))
    return statistics, stats


def _test_transforms(model, dset):
    sample = dset.get_sample("student", "long", "female", 0)
    augmentation = "flat_f0"
    # augmentation = "only_f0"
    # augmentation = "shift_f0"
    # augmentation = "flat_intensity"
    if augmentation == "flat_f0":
        transform = CT.FlatPitch()
        # sample = create_flat_batch(sample)
        sample = create_transform_batch(sample, transform)
    elif augmentation == "only_f0":
        transform = CT.LowPass()
        # sample = create_flat_batch(sample)
        sample = create_transform_batch(sample, transform)
    elif augmentation == "shift_f0":
        transform = CT.ShiftPitch()
        # sample = shift_pitch_batch(sample, factor)
        sample = create_transform_batch(sample, transform)
    elif augmentation == "flat_intensity":
        transform = CT.FlatIntensity(vad_hz=model.frame_hz)
        # sample = NeutralIntensityCallback(vad_hz=model.frame_hz).neutral_batch(sample)
        sample = create_transform_batch(sample, transform)
    loss, out, prob, sample = model.output(sample)
    waveform = sample["waveform"]
    p_ns = prob["p"][0, :, 0]
    w = sample["words"][0]
    s = sample["starts"][0]
    fig, ax = plot_sample(p_ns, waveform, words=w, starts=s)
    plt.show()


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()

    # ch_root = "assets/PaperB/checkpoints"
    # checkpoint = join(ch_root, "cpc_48_20hz_2ucueis8.ckpt")
    # checkpoint = join(ch_root, "cpc_48_50hz_15gqq5s5.ckpt")
    # checkpoint = join(ch_root, "cpc_48_100hz_3mkvq5fk.ckpt")
    model, dset = load_model_dset(args.checkpoint)
    checkpoint_name = basename(args.checkpoint).replace(".ckpt", "")
    statistics, stats = evaluation(model, dset, checkpoint_name)
