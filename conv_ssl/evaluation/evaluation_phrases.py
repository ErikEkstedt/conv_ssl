import torch
import torchaudio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import librosa
from pathlib import Path
from os.path import join, basename, isdir
from copy import deepcopy
from tqdm import tqdm
from os.path import join
from typing import Tuple

from conv_ssl.model import VPModel
from conv_ssl.utils import everything_deterministic
from conv_ssl.utils import write_json
import conv_ssl.transforms as CT
from conv_ssl.augmentations import praat_intensity, torch_to_praat_sound

from conv_ssl.evaluation.phrases.duration import EXAMPLE_TO_TARGET_WORD, get_word_times
from conv_ssl.evaluation.phrases.phrase_dataset import PhraseDataset
from conv_ssl.evaluation.phrases.phrases_duration_process import raw_sample_to_sample

# mpl.use("Agg")

everything_deterministic()


def plot_next_speaker(p_ns, ax, color=["b", "orange"], alpha=0.6, fontsize=12):
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
    ax.set_xlim([0, len(p_ns)])
    ax.set_xticks([])
    ax.set_yticks([0.25, 0.75], ["SHIFT", "HOLD"], fontsize=fontsize)
    ax.set_ylim([0, 1])
    ax.hlines(y=0.5, xmin=0, xmax=len(p_ns), linestyle="dashed", color="k")
    return ax


def plot_sample_data(sample, sample_rate=16000, ax=None, fontsize=12, plot=False):
    snd = torch_to_praat_sound(sample["waveform"].squeeze(1), sample_rate)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    pitch_values[pitch_values == 0] = np.nan
    xmin, xmax = snd.xs().min(), snd.xs().max()
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=800, hop_length=160, n_mels=80
    )(sample["waveform"])[0]
    # print("melspec: ", tuple(melspec.shape))
    # melspec = librosa.power_to_db(
    #     librosa.feature.melspectrogram(
    #         y=deepcopy(sample["waveform"]).squeeze(1).numpy(),
    #         sr=sample_rate,
    #         n_fft=800,
    #         hop_length=160,
    #     ),
    #     ref=np.max,
    # )[0]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(3, 1, figsize=(6, 5))

    # Waveform
    ax[0].plot(snd.xs(), snd.values.T, alpha=0.4)
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_ylabel("waveform", fontsize=fontsize)

    target_word = None
    if "example" in sample:
        target_word = EXAMPLE_TO_TARGET_WORD[sample["example"]]
    # bounds = get_word_times(target_word, sample)
    # end_time = bounds[0][1]

    SCP_line_x = None
    if "words" in sample:
        # Plot text on top of waveform
        y_min = -0.8
        y_max = 0.8
        diff = y_max - y_min
        steps = 4
        for ii, (start_time, end_time, word) in enumerate(sample["words"]):
            yy = y_min + diff * (ii % steps) / steps
            mid = start_time + 0.5 * (end_time - start_time)

            # previous word was SCP
            c = "k"
            llw = 1
            if target_word is not None:
                if (
                    sample["words"][ii - 1][-1] == target_word
                    and sample["size"] == "long"
                ):
                    c = "r"
                    llw = 2
                    SCP_line_x = start_time

            ax[0].text(
                x=mid,
                y=yy,
                s=word,
                fontsize=12,
                horizontalalignment="center",
            )
            ax[0].vlines(
                start_time,
                ymin=-1,
                ymax=1,
                linestyle="dashed",
                linewidth=llw,
                color=c,
                alpha=0.8,
            )

        # final end of word
        ax[0].vlines(
            sample["words"][-1][1],
            ymin=-1,
            ymax=1,
            # linestyle="dashed",
            linewidth=2,
            color="r",
            alpha=0.8,
        )

    # SPEC
    # img = librosa.display.specshow(
    #     melspec[0].numpy(),
    #     x_axis="time",
    #     y_axis="mel",
    #     sr=sample_rate,
    #     hop_length=160,
    #     fmax=8000,
    #     ax=ax[1],
    # )
    ax[1].imshow(melspec, aspect="auto", origin="lower")
    # ax[1].set_xlim([0, melspec.shape[-1]])
    ymin, ymax = ax[1].get_ylim()
    if "words" in sample:
        ax[1].vlines(
            sample["words"][-1][1],
            ymin=ymin,
            ymax=ymax,
            # linestyle="dashed",
            linewidth=2,
            color="r",
        )

    if SCP_line_x is not None:
        ax[1].vlines(
            SCP_line_x,
            ymin=ymin,
            ymax=ymax,
            linestyle="dashed",
            linewidth=2,
            color="r",
        )
    ax[1].set_xticks([])
    ax[1].set_xlabel("")
    ax[1].set_ylabel("Mel (Hz)", fontsize=fontsize)
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
    ymin, ymax = ax[2].get_ylim()
    if "words" in sample:
        ax[2].vlines(
            sample["words"][-1][1],
            ymin=ymin,
            ymax=ymax,
            # linestyle="dashed",
            linewidth=2,
            color="r",
        )
    if SCP_line_x is not None:
        ax[2].vlines(
            SCP_line_x,
            ymin=ymin,
            ymax=ymax,
            linestyle="dashed",
            linewidth=2,
            color="r",
        )
    ax[2].yaxis.tick_right()
    ax[2].set_ylabel("F0 (Hz)", fontsize=fontsize)

    plt.subplots_adjust(
        left=0.1, bottom=0.05, right=0.9, top=0.95, wspace=None, hspace=0.09
    )

    if plot:
        plt.pause(0.1)
    return fig, ax, SCP_line_x


def plot_sample(
    p_ns, sample, sample_rate=16000, frame_hz=50, fontsize=12, ax=None, plot=False
) -> Tuple[plt.Figure, plt.Axes]:
    fig = None
    if ax is None:
        fig, ax = plt.subplots(4, 1, figsize=(6, 5))

    # plot sample data ax[0], ax[1], ax[2]
    _, ax, scp_line_x = plot_sample_data(
        sample, sample_rate=sample_rate, ax=ax, fontsize=fontsize
    )

    # Next speaker probs
    _ = plot_next_speaker(p_ns, ax=ax[3], fontsize=fontsize)

    if "words" in sample:
        end_frame = round(sample["words"][-1][1] * frame_hz)
        ax[3].vlines(end_frame, ymin=-1, ymax=1, linewidth=2, color="r")

    if scp_line_x is not None:
        scp_frame = round(scp_line_x * frame_hz)
        ax[3].vlines(
            scp_frame, ymin=-1, ymax=1, linewidth=2, linestyle="dashed", color="r"
        )

    plt.subplots_adjust(
        left=0.1, bottom=0.05, right=0.9, top=0.95, wspace=None, hspace=0.09
    )
    if plot:
        plt.pause(0.1)
    return fig, ax


def load_model_dset(checkpoint, phrase_path):
    if isdir(phrase_path):
        phrase_path = join(phrase_path, "phrases.json")

    model = VPModel.load_from_checkpoint(checkpoint)
    model = model.eval()
    if torch.cuda.is_available():
        _ = model.to("cuda")
    print("model sample_rate: ", model.sample_rate)
    print("model mono: ", model.mono)
    print("model VAP.horizon: ", model.VAP.horizon)

    dset = PhraseDataset(
        phrase_path,
        vad_hz=model.frame_hz,
        sample_rate=model.sample_rate,
        vad_horizon=model.VAP.horizon,
        vad_history=model.conf["data"]["vad_history"],
        vad_history_times=model.conf["data"]["vad_history_times"],
    )
    return model, dset


def get_region_probs(
    p, sample, eot_or_scp, speaker=1, predictive_region=0.5, frame_hz=50
):
    if eot_or_scp.lower() == "eot":
        _, end_time, _ = sample["words"][-1]
    elif eot_or_scp.lower() == "scp":
        target_word = EXAMPLE_TO_TARGET_WORD[sample["example"]]
        bounds = get_word_times(target_word, sample)
        end_time = bounds[0][1]
    else:
        raise NotImplementedError(f"{eot_or_scp} not in ['eot', 'scp']")

    last_frame = round(end_time * frame_hz)
    pre_frames = round(predictive_region * frame_hz)
    pre_cutoff_frames = last_frame - pre_frames

    # Changed to only measure inside of word/utterance
    hold = p[0, :pre_cutoff_frames, speaker]
    predictive = p[0, pre_cutoff_frames + 1 : last_frame - 2, speaker]
    reactive = p[0, last_frame - 2 : last_frame + 1, speaker]
    return hold, predictive, reactive


def evaluationOLD(model, dset, checkpoint_name, savepath, predictive_region=0.2):
    """
    Aggregate evaluation

    * save global stats
    * save all figures
    """

    def identity(waveform, vad=None):
        """Used as a transform when 'regular'"""
        return waveform.squeeze(0)

    transforms = {
        "flat_f0": CT.FlatPitch(),
        "only_f0": CT.LowPass(),
        "shift_f0": CT.ShiftPitch(),
        "flat_intensity": CT.FlatIntensity(vad_hz=model.frame_hz),
        "avg_duration": None,
        "regular": identity,
    }

    # Savepaths
    savepath += f"_pred{predictive_region}"
    root = join(savepath, checkpoint_name)
    fig_root, wav_root = join(root, "figs"), join(root, "audio")
    Path(fig_root).mkdir(parents=True, exist_ok=True)
    Path(wav_root).mkdir(parents=True, exist_ok=True)
    print("root: ", root)
    print("fig root: ", fig_root)
    print("wav root: ", wav_root)

    all_stats = {}
    pbar = tqdm(range(len(dset)), desc="Phrases evaluation (slow b/c non-batch)")
    for example, v in dset.data.items():
        stats_tmp = {"short": {}, "long": {}}
        for short_long, vv in v.items():
            for gender, sample_list in vv.items():
                for nidx in range(len(sample_list)):
                    fig_dir = join(fig_root, example, short_long, gender)
                    wav_dir = join(wav_root, example, short_long, gender)
                    Path(fig_dir).mkdir(parents=True, exist_ok=True)
                    Path(wav_dir).mkdir(parents=True, exist_ok=True)

                    # Loop over augmentations/permutations and store figs and stats
                    input()
                    for augmentation, transform in transforms.items():
                        try:
                            sample = dset.get_sample(example, short_long, gender, nidx)
                        except:
                            print("Could not load data")
                            continue
                        name = f"{example}_{gender}_{short_long}_{sample['tts']}"
                        if augmentation == "avg_duration":
                            # we need to change the phone/word timing information
                            # when changing the durations
                            sample = raw_sample_to_sample(sample, dset)
                            _, _, prob, sample = model.output(sample)
                        else:
                            # n_frames = sample["vad_history"].shape[1]
                            # vad = sample["vad"][:, :n_frames]
                            sample["waveform"] = transform(
                                sample["waveform"].unsqueeze(1), vad=sample["vad"]
                            )
                            _, _, prob, sample = model.output(sample)

                        # Save Figure
                        try:
                            fig, _ = plot_sample(prob["p"][0, :, 0], sample)
                            fig.savefig(join(fig_dir, name + f"_{augmentation}.png"))
                            print("+ Worked: ", augmentation)
                        except Exception as e:
                            print("- Broken: ", augmentation, sample["audio_path"])
                            continue

                        # save waveform
                        wavpath = join(wav_dir, name + f"_{augmentation}.wav")
                        torchaudio.save(
                            wavpath, sample["waveform"], sample_rate=model.sample_rate
                        )

                        # Save Statistics
                        if augmentation not in stats_tmp[short_long]:
                            stats_tmp[short_long][augmentation] = {
                                "short_completion": {"pre": [], "end": [], "post": []},
                                "last_completion": {"pre": [], "end": [], "post": []},
                            }

                        # print(short_long.upper())
                        pre, end, post = get_SCP_regions_probs(
                            prob["p"],
                            sample,
                            predictive_region=predictive_region,
                            frame_hz=model.frame_hz,
                        )
                        stats_tmp[short_long][augmentation]["short_completion"][
                            "pre"
                        ].append(pre)
                        stats_tmp[short_long][augmentation]["short_completion"][
                            "end"
                        ].append(end)
                        stats_tmp[short_long][augmentation]["short_completion"][
                            "post"
                        ].append(post)

                        # End
                        pre_last, end_last, post_last = get_EOT_region_probs(
                            prob["p"],
                            sample,
                            predictive_region=predictive_region,
                            frame_hz=model.frame_hz,
                        )
                        stats_tmp[short_long][augmentation]["last_completion"][
                            "pre"
                        ].append(pre_last)
                        stats_tmp[short_long][augmentation]["last_completion"][
                            "end"
                        ].append(end_last)
                        stats_tmp[short_long][augmentation]["last_completion"][
                            "post"
                        ].append(post_last)

                        # Close figures
                        plt.close("all")
                    # pbar.update()
        all_stats[example] = stats_tmp

    # Global Stats
    score = {}
    for example, stats_tmp in all_stats.items():
        statistics = {}
        for long_short in ["long", "short"]:
            statistics[long_short] = {}
            for (
                augmentation,
                comp_stats,
            ) in stats_tmp[long_short].items():
                if augmentation not in statistics[long_short]:
                    statistics[long_short][augmentation] = {}
                for completion in ["short_completion", "last_completion"]:
                    pre = 0
                    pre_frames = 0
                    end = 0
                    end_frames = 0
                    post = 0
                    post_frames = 0
                    for s in comp_stats[completion]["pre"]:
                        pre += s.sum()
                        pre_frames += len(s)
                    for s in comp_stats[completion]["end"]:
                        end += s.sum()
                        end_frames += len(s)
                    for s in comp_stats[completion]["post"]:
                        post += s.sum()
                        post_frames += len(s)
                    pre /= pre_frames
                    end /= end_frames
                    post /= post_frames
                    pre = pre.item()
                    end = 1 - end.item()  # hold -> shift probs
                    avg = (end + pre) / 2
                    post = 1 - post.item()
                    statistics[long_short][augmentation][completion] = {
                        "pre": pre,
                        "end": end,
                        "avg": avg,
                        "post": post,
                    }
        score[example] = statistics

    write_json(score, join(root, "score.json"))
    write_json(model.conf, join(root, "config.json"))
    print("Saved stats -> ", join(root, "score.json"))
    print("Saved config -> ", join(root, "config.json"))
    return score


def extract_scores(all_stats):
    score = {}
    for example, stats_tmp in all_stats.items():
        example_stats = {}
        for LS in ["long", "short"]:
            example_stats[LS] = {}
            for perm, comp_stats in stats_tmp[LS].items():
                if perm not in example_stats[LS]:
                    example_stats[LS][perm] = {}
                for completion in ["SCP", "EOT"]:
                    if completion == "EOT" and LS == "short":
                        continue

                    hold, hold_frames = 0, 0
                    predictive, predictive_frames = 0, 0
                    reactive, reactive_frames = 0, 0
                    for s in comp_stats[completion]["hold"]:
                        hold += s.sum()
                        hold_frames += len(s)
                    for s in comp_stats[completion]["predictive"]:
                        predictive += s.sum()
                        predictive_frames += len(s)
                    for s in comp_stats[completion]["reactive"]:
                        reactive += s.sum()
                        reactive_frames += len(s)

                    # Average over frames
                    hold = hold.item() / hold_frames
                    predictive = predictive.item() / predictive_frames
                    reactive = reactive.item() / reactive_frames

                    example_stats[LS][perm][completion] = {
                        "hold": hold,
                        "predictive": predictive,
                        "reactive": reactive,
                    }
        score[example] = example_stats

    return score


def evaluation(model, dset, checkpoint_name, savepath=None, predictive_region=0.2):
    def identity(waveform, vad=None):
        """Used as a transform when 'regular'"""
        return waveform.squeeze(0)

    transforms = {
        "flat_f0": CT.FlatPitch(sample_rate=dset.sample_rate),
        "only_f0": CT.LowPass(sample_rate=dset.sample_rate),
        "shift_f0": CT.ShiftPitch(sample_rate=dset.sample_rate),
        "flat_intensity": CT.FlatIntensity(
            vad_hz=model.frame_hz, sample_rate=dset.sample_rate
        ),
        "avg_duration": None,
        "regular": identity,
    }

    # Savepaths
    fig_root, wav_root = None, None
    if savepath is not None:
        savepath += f"_pred{predictive_region}"
        root = join(savepath, checkpoint_name)
        fig_root, wav_root = join(root, "figs"), join(root, "audio")
        Path(fig_root).mkdir(parents=True, exist_ok=True)
        Path(wav_root).mkdir(parents=True, exist_ok=True)
        print("root: ", root)
        print("fig root: ", fig_root)
        print("wav root: ", wav_root)

    all_stats = {}
    pbar = tqdm(range(len(dset)), desc="Phrases evaluation (slow b/c non-batch)")
    for example, v in dset.data.items():
        stats_tmp = {"short": {}, "long": {}}
        for SL, vv in v.items():
            for gender, sample_list in vv.items():
                for nidx in range(len(sample_list)):
                    pbar.update()
                    for perm, transform in transforms.items():
                        sample = dset.get_sample(example, SL, gender, nidx)
                        name = f"{example}_{gender}_{SL}_{sample['tts']}"
                        if perm == "avg_duration":
                            # we need to change the phone/word timing information
                            # when changing the durations
                            sample = raw_sample_to_sample(sample, dset)
                            if sample["waveform"].ndim > 2:
                                sample["waveform"] = sample["waveform"].squeeze(0)
                        else:
                            sample["waveform"] = transform(
                                waveform=sample["waveform"], vad=sample["vad"]
                            )

                        # print(
                        #     f"{perm} sample['waveform']: ",
                        #     tuple(sample["waveform"].shape),
                        #     sample["waveform"].ndim,
                        # )
                        _, _, prob, sample = model.output(sample)

                        # Create and save figures
                        if fig_root is not None:
                            fig_dir = join(fig_root, example, SL, gender)
                            Path(fig_dir).mkdir(parents=True, exist_ok=True)
                            fig, _ = plot_sample(prob["p"][0, :, 0], sample)
                            fig.savefig(join(fig_dir, name + f"_{perm}.png"))

                        # Save waveform
                        # wav_dir = join(wav_root, example, SL, gender)
                        # Path(wav_dir).mkdir(parents=True, exist_ok=True)
                        # wavpath = join(wav_dir, name + f"_{perm}.wav")
                        # torchaudio.save(
                        #     wavpath, sample["waveform"], sample_rate=model.sample_rate
                        # )

                        # Save Statistics
                        if perm not in stats_tmp[SL]:
                            stats_tmp[SL][perm] = {
                                "SCP": {"hold": [], "predictive": [], "reactive": []},
                            }
                            if SL == "long":
                                stats_tmp[SL][perm]["EOT"] = {
                                    "hold": [],
                                    "predictive": [],
                                    "reactive": [],
                                }

                        # Finds the regions at the Short-Completion-Point which is also the end of the utterance for the short-samples
                        hold, predictive, reactive = get_region_probs(
                            prob["p"],
                            sample,
                            eot_or_scp="SCP",
                            predictive_region=predictive_region,
                            frame_hz=model.frame_hz,
                            speaker=0,
                        )
                        stats_tmp[SL][perm]["SCP"]["hold"].append(hold)
                        stats_tmp[SL][perm]["SCP"]["predictive"].append(predictive)
                        stats_tmp[SL][perm]["SCP"]["reactive"].append(reactive)

                        # Finds the regions at the End-of-Turn which
                        if SL == "long":
                            hold, predictive, reactive = get_region_probs(
                                prob["p"],
                                sample,
                                eot_or_scp="EOT",
                                predictive_region=predictive_region,
                                frame_hz=model.frame_hz,
                                speaker=0,
                            )
                            stats_tmp[SL][perm]["EOT"]["hold"].append(hold)
                            stats_tmp[SL][perm]["EOT"]["predictive"].append(predictive)
                            stats_tmp[SL][perm]["EOT"]["reactive"].append(reactive)

                        # Close figures
                        plt.close("all")
        all_stats[example] = stats_tmp

    torch.save(all_stats, "all_stats.pt")
    score = extract_scores(all_stats)
    return score, all_stats


def _delete_this():

    all_stats = torch.load("all_stats.pt")
    print(all_stats["student"]["short"]["flat_f0"]["SCP"]["hold"])
    score = extract_scores(all_stats)
    print(score["psychology"]["long"]["regular"]["SCP"])
    for example in score.keys():
        print("hold: ", score[example]["long"]["regular"]["EOT"]["hold"])
        print("predictive: ", score[example]["long"]["regular"]["EOT"]["predictive"])
        print("reactive: ", score[example]["long"]["regular"]["EOT"]["reactive"])
        print()


def _test_transforms(model, dset):
    import sounddevice as sd

    ch_root = "assets/PaperB/checkpoints"
    # checkpoint = join(ch_root, "cpc_48_20hz_2ucueis8.ckpt")
    checkpoint = join(ch_root, "cpc_48_50hz_15gqq5s5.ckpt")
    # checkpoint = join(ch_root, "cpc_48_100hz_3mkvq5fk.ckpt")
    model, dset = load_model_dset(checkpoint)
    augmentation = "regular"
    # augmentation = "flat_f0"
    # augmentation = "only_f0"
    # augmentation = "shift_f0"
    # augmentation = "flat_intensity"
    sample = dset.get_sample("basketball", "long", "male", 2)
    if augmentation == "flat_f0":
        sample["waveform"] = CT.FlatPitch()(
            sample["waveform"].unsqueeze(0), sample["vad"]
        )
    elif augmentation == "only_f0":
        sample["waveform"] = CT.LowPass()(
            sample["waveform"].unsqueeze(0), sample["vad"]
        )
    elif augmentation == "shift_f0":
        sample["waveform"] = CT.ShiftPitch()(
            sample["waveform"].unsqueeze(0), sample["vad"]
        )
    elif augmentation == "flat_intensity":
        sample["waveform"] = CT.FlatIntensity(vad_hz=model.frame_hz)(
            sample["waveform"].unsqueeze(0), sample["vad"]
        )
    loss, out, prob, sample = model.output(sample)
    fig, ax = plot_sample(prob["p"][0, :, 0], sample, frame_hz=model.frame_hz)
    sd.play(sample["waveform"].squeeze(), samplerate=16000)
    plt.show()


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Model checkpoint path",
        default="example/cpc_48_50hz_15gqq5s5.ckpt",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset_phrases/phrases.json",
        help="Path (relative) to phrase-dataset or directly to the 'phrases.json' file used in 'PhraseDataset'",
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default="runs_evaluation/phrases2",
        help="Path to results directory",
    )
    args = parser.parse_args()
    model, dset = load_model_dset(args.checkpoint_path, phrase_path=args.dataset)

    checkpoint_name = basename(args.checkpoint_path).replace(".ckpt", "")
    score, all_stats = evaluation(
        model=model, dset=dset, checkpoint_name=checkpoint_name, savepath=args.savepath
    )
