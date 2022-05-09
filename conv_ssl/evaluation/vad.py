from datasets_turntaking.utils import load_waveform

try:
    from pyannote.audio import Pipeline
except ImportError as e:
    print(
        """
        Install pyannote
        ```bash
            conda create -n pyannote python=3.8
            conda activate pyannote
            conda install pytorch torchaudio -c pytorch
            pip install https://github.com/pyannote/pyannote-audio/archive/develop.zip
        ```
            """
    )


class VadExtractor:
    def __init__(self):
        self.vad_pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection"
        )

    def __call__(self, y, sample_rate):
        vad_list = [[], []]
        for channel in range(y.shape[0]):
            audio_in_memory = {
                "waveform": y[channel : channel + 1],
                "sample_rate": sample_rate,
            }
            out = self.vad_pipeline(audio_in_memory)
            for segment in out.get_timeline():
                vad_list[channel].append(
                    [round(segment.start, 2), round(segment.end, 2)]
                )
        return vad_list


if __name__ == "__main__":
    wav_path = "assets/phrases/audio/basketball_long_female_en-US-Wavenet-C.wav"
    y, sr = load_waveform(wav_path, sample_rate=16000)
    vadder = VadExtractor()
    vad_list = vadder(y, sample_rate=sr)
    print(vad_list)
