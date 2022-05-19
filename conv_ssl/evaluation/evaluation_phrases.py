import torch
import torchaudio
from pathlib import Path
from os.path import join, basename
from copy import deepcopy
from tqdm import tqdm
from os.path import join

from conv_ssl.evaluation.duration import EXAMPLE_TO_TARGET_WORD, get_word_times
from conv_ssl.evaluation.phrase_dataset import PhraseDataset, plot_sample_data
from conv_ssl.utils import write_json
import conv_ssl.transforms as CT
from conv_ssl.evaluation.phrases_duration_process import raw_sample_to_sample

from conv_ssl.model import VPModel
from conv_ssl.utils import everything_deterministic

import matplotlib.pyplot as plt


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


def plot_sample(
    p_ns, sample, sample_rate=16000, frame_hz=50, fontsize=12, ax=None, plot=False
):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(4, 1, figsize=(6, 5))

    # plot sample data ax[0], ax[1], ax[2]
    _, ax, scp_line_x = plot_sample_data(
        sample, sample_rate=sample_rate, ax=ax, fontsize=fontsize
    )

    # Next speaker probs
    _ = plot_next_speaker(p_ns, ax=ax[3], fontsize=fontsize)

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


def get_hold_prob_end_of_utt(p, sample, pre_cutoff=0.5, post_cutoff=0.2, frame_hz=50):
    _, end_time, _ = sample["words"][-1]

    last_frame = round(end_time * frame_hz)
    pre_frames = round(pre_cutoff * frame_hz)
    # post_frames = round(post_cutoff * frame_hz)

    pre_cutoff_frames = last_frame - pre_frames
    # post_cutoff_frames = last_frame + post_frames
    # pre = p[0, :pre_cutoff_frames, 0]
    # end = p[0, pre_cutoff_frames + 1 : last_frame + 1, 0]
    # post = p[0, last_frame + 1 : post_cutoff_frames + 1, 0]

    # Changed to only measure inside of word/utterance
    pre = p[0, :pre_cutoff_frames, 0]
    end = p[0, pre_cutoff_frames + 1 : last_frame - 2, 0]
    post = p[0, last_frame - 2 : last_frame + 1, 0]

    # print("LAST COMPLETION")
    # print(sample["words"])
    # print("Last: ", end_time, w)
    # print("pre_frames: ", pre_frames)
    # print("last_frame: ", last_frame)
    # print("post_frames: ", post_frames)
    # print("pre: ", pre_cutoff_frames, pre.mean())
    # print("last: ", last_frame, end.mean())
    # print("post: ", post_cutoff_frames, post.mean())
    # input()

    return pre, end, post


def get_pre_end_hold_probs(p, sample, pre_cutoff=0.5, post_cutoff=0.2, frame_hz=50):
    target_word = EXAMPLE_TO_TARGET_WORD[sample["example"]]
    bounds = get_word_times(target_word, sample)
    end_time = bounds[0][1]

    last_frame = round(end_time * frame_hz)
    pre_cutoff_frames = last_frame - round(pre_cutoff * frame_hz)
    # post_cutoff_frames = last_frame + round(post_cutoff * frame_hz)
    # pre = p[0, :pre_cutoff_frames, 0]
    # end = p[0, pre_cutoff_frames + 1 : last_frame + 1, 0]
    # post = p[0, last_frame + 1 : post_cutoff_frames + 1, 0]

    # Changed to only measure inside of word/utterance
    pre = p[0, :pre_cutoff_frames, 0]
    end = p[0, pre_cutoff_frames + 1 : last_frame - 2, 0]
    post = p[0, last_frame - 2 : last_frame + 1, 0]
    # print("SHORT COMPLETION")
    # print("pre: ", pre_cutoff_frames, pre.mean())
    # print("last: ", end_frame, end.mean())
    # print("post: ", post_cutoff_frames, post.mean())
    return pre, end, post


def identity(waveform, vad=None):
    """Used as a transform when 'regular'"""
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
        "avg_duration": None,
        "regular": identity,
    }

    # savepath = "assets/PaperB/eval_phrases_test"
    # ch_root = "assets/PaperB/checkpoints"
    # # checkpoint = join(ch_root, "cpc_48_20hz_2ucueis8.ckpt")
    # checkpoint = join(ch_root, "cpc_48_50hz_15gqq5s5.ckpt")
    # checkpoint_name = basename(checkpoint).replace(".ckpt", "")
    # # checkpoint = join(ch_root, "cpc_48_100hz_3mkvq5fk.ckpt")
    # model, dset = load_model_dset(checkpoint)

    pre_cutoff = 0.2
    post_cutoff = 0.2
    savepath += "_pre02"
    root = join(savepath, checkpoint_name)
    fig_root = join(root, "figs")
    wav_root = join(root, "audio")
    Path(fig_root).mkdir(parents=True, exist_ok=True)
    Path(wav_root).mkdir(parents=True, exist_ok=True)
    print("root: ", root)
    print("fig root: ", fig_root)
    print("wav root: ", wav_root)

    all_stats = {}
    pbar = tqdm(range(len(dset)))
    for example, v in dset.data.items():
        stats = {"short": {}, "long": {}}
        for short_long, vv in v.items():
            for gender, sample_list in vv.items():
                for nidx in range(len(sample_list)):
                    fig_dir = join(fig_root, example, short_long, gender)
                    wav_dir = join(wav_root, example, short_long, gender)
                    Path(fig_dir).mkdir(parents=True, exist_ok=True)
                    Path(wav_dir).mkdir(parents=True, exist_ok=True)

                    # Get Original Sample
                    sample = dset.get_sample(example, short_long, gender, nidx)
                    orig_sample = deepcopy(sample)
                    orig_waveform = deepcopy(sample["waveform"].unsqueeze(0))

                    name = f"{example}_{gender}_{short_long}_{sample['tts']}"

                    for augmentation, transform in transforms.items():
                        if augmentation == "avg_duration":
                            # we need to change the phone/word timing information
                            # when changing the durations
                            sample = raw_sample_to_sample(orig_sample, dset)
                            _, _, prob, sample = model.output(sample)
                        else:
                            n_frames = orig_sample["vad_history"].shape[1]
                            vad = orig_sample["vad"][:, :n_frames]
                            orig_sample["waveform"] = transform(orig_waveform, vad=vad)
                            _, _, prob, sample = model.output(orig_sample)

                        # Save Figure
                        fig, _ = plot_sample(prob["p"][0, :, 0], sample)
                        fig.savefig(join(fig_dir, name + f"_{augmentation}.png"))

                        # save waveform
                        wavpath = join(wav_dir, name + f"_{augmentation}.wav")
                        torchaudio.save(
                            wavpath, sample["waveform"], sample_rate=model.sample_rate
                        )

                        # Save Statistics
                        if augmentation not in stats[short_long]:
                            stats[short_long][augmentation] = {
                                "short_completion": {"pre": [], "end": [], "post": []},
                                "last_completion": {"pre": [], "end": [], "post": []},
                            }

                        # print(short_long.upper())
                        pre, end, post = get_pre_end_hold_probs(
                            prob["p"],
                            sample,
                            pre_cutoff=pre_cutoff,
                            post_cutoff=post_cutoff,
                            frame_hz=model.frame_hz,
                        )
                        stats[short_long][augmentation]["short_completion"][
                            "pre"
                        ].append(pre)
                        stats[short_long][augmentation]["short_completion"][
                            "end"
                        ].append(end)
                        stats[short_long][augmentation]["short_completion"][
                            "post"
                        ].append(post)

                        # End
                        pre_last, end_last, post_last = get_hold_prob_end_of_utt(
                            prob["p"],
                            sample,
                            pre_cutoff=pre_cutoff,
                            post_cutoff=post_cutoff,
                            frame_hz=model.frame_hz,
                        )
                        stats[short_long][augmentation]["last_completion"][
                            "pre"
                        ].append(pre_last)
                        stats[short_long][augmentation]["last_completion"][
                            "end"
                        ].append(end_last)
                        stats[short_long][augmentation]["last_completion"][
                            "post"
                        ].append(post_last)

                        # Close figures
                        plt.close("all")
                    pbar.update()
        all_stats[example] = stats

    # Global Stats
    all_statistics = {}
    for example, stats in all_stats.items():
        statistics = {}
        for long_short in ["long", "short"]:
            statistics[long_short] = {}
            for (
                augmentation,
                comp_stats,
            ) in stats[long_short].items():
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
        all_statistics[example] = statistics

    torch.save(all_stats, join(root, "stats.pt"))
    write_json(all_statistics, join(root, "score.json"))
    print("Saved stats -> ", join(root, "score.json"))
    return statistics, stats


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
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()
    # ch_root = "assets/PaperB/checkpoints"
    # checkpoint = join(ch_root, "cpc_48_20hz_2ucueis8.ckpt")
    # checkpoint = join(ch_root, "cpc_48_50hz_15gqq5s5.ckpt")
    # checkpoint = join(ch_root, "cpc_48_100hz_3mkvq5fk.ckpt")
    model, dset = load_model_dset(args.checkpoint)
    checkpoint_name = basename(args.checkpoint).replace(".ckpt", "")
    statistics, stats = evaluation(model, dset, checkpoint_name)
