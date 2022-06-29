import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join, basename
import matplotlib.pyplot as plt
import librosa
from librosa import display

from conv_ssl.augmentations import torch_to_praat_sound
from conv_ssl.evaluation.duration import read_text_grid, EXAMPLE_TO_TARGET_WORD
from conv_ssl.utils import read_json
from datasets_turntaking.utils import (
    load_waveform,
    get_audio_info,
    time_to_frames,
)
from vap_turn_taking.utils import vad_list_to_onehot, get_activity_history


TEXT_GRID_ROOT = "assets/phrases_beta/alignment"


def audio_path_text_grid_path(path):
    name = basename(path).replace(".wav", ".TextGrid")
    return join(TEXT_GRID_ROOT, name)


def words_to_vad(words):
    vad_list = []
    for s, e, w in words:
        vad_list.append([s, e])
    return [vad_list, []]


def plot_sample_data(sample, sample_rate=16000, ax=None, fontsize=12, plot=False):
    snd = torch_to_praat_sound(sample["waveform"], sample_rate)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    pitch_values[pitch_values == 0] = np.nan
    xmin, xmax = snd.xs().min(), snd.xs().max()
    melspec = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=sample["waveform"].numpy(), sr=sample_rate, n_fft=800, hop_length=160
        ),
        ref=np.max,
    )[0]

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
    img = librosa.display.specshow(
        melspec,
        x_axis="time",
        y_axis="mel",
        sr=sample_rate,
        hop_length=160,
        fmax=8000,
        ax=ax[1],
    )
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


class PhraseDataset(Dataset):
    def __init__(
        self,
        phrase_path,
        # AUDIO #################################
        sample_rate=16000,
        audio_mono=True,
        audio_duration=10,
        audio_normalize=True,
        # VAD #################################
        vad=True,
        vad_hz=100,
        vad_horizon=2,
        vad_history=False,
        vad_history_times=[60, 30, 10, 5],
        **kwargs,
    ):
        super().__init__()
        self.data = read_json(phrase_path)
        self.indices = self.map_phrases_to_idx()

        # Audio (waveforms)
        self.sample_rate = sample_rate
        self.audio_mono = audio_mono
        self.audio_duration = audio_duration
        self.audio_normalize = audio_normalize
        self.audio_normalize_threshold = 0.05

        # VAD parameters
        self.vad = vad  # use vad or not
        self.vad_hz = vad_hz
        self.vad_hop_time = 1.0 / vad_hz

        # Vad prediction labels
        self.horizon_time = vad_horizon
        self.vad_horizon = time_to_frames(vad_horizon, hop_time=self.vad_hop_time)

        # Vad history
        self.vad_history = vad_history
        self.vad_history_times = vad_history_times
        self.vad_history_frames = (
            (torch.tensor(vad_history_times) / self.vad_hop_time).long().tolist()
        )

    def map_phrases_to_idx(self):
        indices = []
        for example, v in self.data.items():
            for long_short, vv in v.items():
                for gender, sample_list in vv.items():
                    for ii in range(len(sample_list)):
                        indices.append([example, long_short, gender, ii])
        return indices

    def __repr__(self):
        s = "Phrase Dataset"
        s += f"\n\tsample_rate: {self.sample_rate}"
        s += f"\n\taudio_mono: {self.audio_mono}"
        s += f"\n\taudio_normalize: {self.audio_normalize}"
        s += f"\n\taudio_normalize_threshold: {self.audio_normalize_threshold}"

        # VAD parameters
        s += f"\n\tvad_hz: {self.vad_hz}"
        s += f"\n\tvad_hop_time: {self.vad_hop_time}"
        s += f"\n\tvad_horizon: {self.vad_horizon}"

        # Vad history
        s += f"\n\tvad_history: {self.vad_history}"
        s += f"\n\tvad_history_times: {self.vad_history_times}"
        s += f"\n\tvad_history_frames: {self.vad_history_frames}"
        s += "\n" + "-" * 40
        return s

    def __len__(self):
        return len(self.indices)

    def _sample_data(self, b, vad_list):
        """
        Get the sample from the dialog

        Returns dict containing:
            waveform,
            dataset_name,
            vad,
            vad_history
        """
        # Loads the dialog waveform (stereo) and normalize/to-mono for each
        # smaller segment in loop below
        waveform, _ = load_waveform(
            b["audio_path"],
            sample_rate=self.sample_rate,
            normalize=self.audio_normalize,
            mono=self.audio_mono,
        )

        # dict to return
        ret = {
            "waveform": waveform,
            "dataset_name": "phrases",
        }

        # VAD-frame of relevant part
        if self.vad:
            duration = get_audio_info(b["audio_path"])["duration"]
            start_frame = 0
            end_frame = time_to_frames(duration, self.vad_hop_time)
            all_vad_frames = vad_list_to_onehot(
                vad_list,
                hop_time=self.vad_hop_time,
                duration=duration,
                channel_last=True,
            )

        ##############################################
        # History
        ##############################################
        if self.vad and self.vad_history:
            # history up until the current features arrive
            vad_history, _ = get_activity_history(
                all_vad_frames,
                bin_end_frames=self.vad_history_frames,
                channel_last=True,
            )
            # ret["vad_history"] = vad_history[start_frame:end_frame].unsqueeze(0)
            # vad history is always defined as speaker 0 activity
            ret["vad_history"] = vad_history[start_frame:end_frame][..., 0].unsqueeze(0)

        ##############################################
        # VAD
        ##############################################
        if self.vad:
            if end_frame + self.vad_horizon > all_vad_frames.shape[0]:
                lookahead = torch.zeros(
                    (self.vad_horizon + 1, 2)
                )  # add horizon after end (silence)
                all_vad_frames = torch.cat((all_vad_frames, lookahead))
            ret["vad"] = all_vad_frames[
                start_frame : end_frame + self.vad_horizon
            ].unsqueeze(0)
        return ret

    def get_sample_data(self, sample, example):
        tg = read_text_grid(audio_path_text_grid_path(sample["audio_path"]))
        vad_list = words_to_vad(tg["words"])
        # Returns: waveform, dataset_name, vad, vad_history
        ret = self._sample_data(sample, vad_list)
        ret["example"] = example
        ret["words"] = tg["words"]
        ret["phones"] = tg["phones"]

        for k, v in sample.items():
            if k in ["vad", "words"]:
                continue
            ret[k] = v
        # ret["gender"] = sample["gender"]
        # ret["size"] = sample["size"]
        # ret["text"] = sample["text"]
        return ret

    def get_sample(self, example, long_short, gender, id):
        sample = self.data[example][long_short][gender][id]
        return self.get_sample_data(sample, example)

    def __getitem__(self, idx):
        example, long_short, gender, nidx = self.indices[idx]
        return self.get_sample(example, long_short, gender, nidx)


if __name__ == "__main__":

    dset = PhraseDataset("assets/phrases_beta/phrases.json")
    sample = dset.get_sample("student", "short", "female", 0)
    for k, v in sample.items():
        print(f"{k}: {type(v)}")

    sample = dset.get_sample("student", "long", "female", 2)
    fig, ax, scp_line = plot_sample_data(sample)
    plt.show()
