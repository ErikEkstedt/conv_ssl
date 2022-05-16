import torch
from torch.utils.data import Dataset
from conv_ssl.utils import read_json
from datasets_turntaking.utils import (
    load_waveform,
    get_audio_info,
    time_to_frames,
)
from vap_turn_taking.utils import vad_list_to_onehot, get_activity_history


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
        # for example, gender_dict in self.data.items():
        #     for gender, sample_list in gender_dict.items():
        #         for ii in range(len(sample_list)):
        #             indices.append([example, gender, ii])
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

    def get_sample_data(self, b, example):
        """Get the sample from the dialog"""
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
            "session": f'{example}_{b["gender"]}_{b["tts"]}',
        }
        ret.update(b)

        # VAD-frame of relevant part
        if self.vad:
            duration = get_audio_info(b["audio_path"])["duration"]
            start_frame = 0
            end_frame = time_to_frames(duration, self.vad_hop_time)
            all_vad_frames = vad_list_to_onehot(
                b["vad"],
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

        ##############################################
        # Include words/starts/size if included
        ##############################################
        if "words" in b:
            ret["words"] = [b["words"]]
        if "starts" in b:
            ret["starts"] = [b["starts"]]
        if "size" in b:
            ret["size"] = [b["size"]]
        return ret

    def get_sample(self, example, long_short, gender, id):
        sample = self.data[example][long_short][gender][id]
        return self.get_sample_data(sample, example)

    def __getitem__(self, idx):
        example, long_short, gender, nidx = self.indices[idx]
        sample = self.data[example][long_short][gender][nidx]
        return self.get_sample_data(sample, example)


if __name__ == "__main__":
    dset = PhraseDataset("assets/phrases_beta/phrases.json")
    sample = dset.get_sample("student", "short", "female", 0)
    for k, v in sample.items():
        print(f"{k}: {type(v)}")
