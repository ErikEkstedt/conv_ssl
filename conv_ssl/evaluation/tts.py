from google.cloud import texttospeech
import google.cloud.texttospeech_v1beta1 as texttospeech_beta
from os.path import join
from os import makedirs
from os import environ
import numpy as np
from typing import Tuple
from tqdm import tqdm

from conv_ssl.utils import read_json, write_json


# Temporary hack
environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = "/home/erik/projects/data/GOOGLE_SPEECH_CREDENTIALS.json"


"""
Docs beta: 
    https://cloud.google.com/python/docs/reference/texttospeech/latest/google.cloud.texttospeech_v1beta1.services.text_to_speech.TextToSpeechClient


"""


class TTS(object):
    genders = ["male", "female", "netral"]
    gender_bias: float = 0.5
    voices: dict

    def _get_voices(self):
        """Performs the list voices request"""
        voices = {"female": [], "male": [], "neutral": []}
        for v in self.client.list_voices(language_code="en").voices:
            if any([t for t in ["GB", "AU", "US"] if t in v.name]):
                if "wavenet" in v.name.lower():
                    if "female" in v.ssml_gender.name.lower():
                        voices["female"].append(v)
                    elif "neutral" in v.name.lower():
                        voices["neutral"].append(v)
                    else:
                        voices["male"].append(v)
        return voices

    def get_ssml_gender(self, gender):
        if gender.lower() == "female":
            ssml_gender = self.texttospeech.SsmlVoiceGender.FEMALE
        elif gender.lower() == "male":
            ssml_gender = self.texttospeech.SsmlVoiceGender.MALE
        else:
            ssml_gender = self.texttospeech.SsmlVoiceGender.NEUTRAL
        return ssml_gender

    def randomize_gender(self):
        return "male" if np.random.random() < self.gender_bias else "female"

    def get_voice(self, gender="random", voice_idx=None):
        if gender == "random":
            gender = self.randomize_gender()

        ssml_gender = self.get_ssml_gender(gender)

        if voice_idx is None:
            voice_idx = np.random.choice(
                np.arange(len(self.voices[gender])), size=(1,)
            )[0]

        voice = self.voices[gender][voice_idx]
        return self.texttospeech.VoiceSelectionParams(
            language_code=voice.name[:5], name=voice.name, ssml_gender=ssml_gender
        )


class TTSGoogle(TTS):
    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()
        self.voices = self._get_voices()
        self.texttospeech = texttospeech
        self.ENCODING = texttospeech.AudioEncoding.LINEAR16

    def tts(self, text, filepath, voice_info) -> None:
        # Build the voice request, select the language code ("en-US") and the ssml
        # voice gender ("neutral")
        voice = self.texttospeech.VoiceSelectionParams(
            language_code=voice_info.name[:5],
            name=voice_info.name,
            ssml_gender=voice_info.ssml_gender,
        )

        # Perform the text-to-speech request on the text input with the selected
        # voice parameters and audio file type
        response = self.client.synthesize_speech(
            input=self.texttospeech.SynthesisInput(text=text),
            voice=voice,
            audio_config=self.texttospeech.AudioConfig(audio_encoding=self.ENCODING),
        )

        if not filepath.endswith(".wav"):
            filepath += ".wav"

        with open(filepath, "wb") as out:
            out.write(response.audio_content)
        return None


class TTSGoogleBeta(TTS):
    def __init__(self, pitch_max=5):
        assert self._check_pitch(pitch_max), "pitch only defined in [-10, 10]"
        self.pitch_max = pitch_max
        self.client = texttospeech_beta.TextToSpeechClient()
        self.texttospeech = texttospeech_beta
        self.voices = self._get_voices()
        self.ENCODING = self.texttospeech.AudioEncoding.LINEAR16

    def randomize_pitch(self):
        return np.random.randint(-self.pitch_max, self.pitch_max)

    def text_to_ssml_marks(self, text):
        ssml = "<speak>"
        for word in text.split():
            ssml += f'<mark name="{word}"/>{word} '
        ssml += "</speak>"
        return ssml

    def _check_pitch(self, pitch):
        if pitch > 10 or pitch < -10:
            return False
        return True

    def tts(self, text, filepath, voice_info, pitch) -> Tuple[list, list]:
        assert self._check_pitch(pitch), "pitch only defined in [-10, 10]"
        voice = self.texttospeech.VoiceSelectionParams(
            language_code=voice_info.name[:5],
            name=voice_info.name,
            ssml_gender=voice_info.ssml_gender,
        )

        ssml = self.text_to_ssml_marks(text)
        audio_config = self.texttospeech.AudioConfig(
            audio_encoding=self.ENCODING, pitch=pitch
        )

        request = self.texttospeech.SynthesizeSpeechRequest(
            input=self.texttospeech.SynthesisInput(ssml=ssml),
            voice=voice,
            audio_config=audio_config,
            enable_time_pointing=["SSML_MARK"],
        )
        response = self.client.synthesize_speech(request=request)

        starts = []
        words = []
        for t in response.timepoints:
            starts.append(t.time_seconds)
            words.append(t.mark_name)

        # Save Audio
        if not filepath.endswith(".wav"):
            filepath += ".wav"

        with open(filepath, "wb") as out:
            out.write(response.audio_content)

        return words, starts


def get_all_lang_speakers(tts, lang="en-US"):
    voice_infos = {"female": [], "male": []}
    for gender in ["female", "male"]:
        for voice_info in tts.voices[gender]:
            if lang in voice_info.name:
                voice_infos[gender].append(voice_info)
    return voice_infos


def extract_phrases(
    phrase_path="conv_ssl/evaluation/phrases.json", savepath="assets/phrases"
):
    tts = TTSGoogle()
    audio_path = join(savepath, "audio")
    file_path = join(savepath, "annotation")
    makedirs(savepath, exist_ok=True)
    makedirs(audio_path, exist_ok=True)
    makedirs(file_path, exist_ok=True)

    def save_audio_anno(example, utter, voice_info, short_long, gender, id):
        name = f"{example}_{short_long}_{gender}_{id}"
        wav_path = join(audio_path, name + ".wav")
        anno_path = join(file_path, name + ".json")
        tts.tts(text=utter, filepath=wav_path, voice_info=voice_info)
        anno = {
            "text": utterances["short"],
            "audio_path": wav_path,
            "gender": voice_info.ssml_gender.name.lower(),
            "tts": voice_info.name,
        }
        write_json(anno, anno_path)
        return anno

    all_us_voice_info = get_all_lang_speakers(tts)
    phrase_dict = read_json(phrase_path)

    phrases = {}
    pbar = tqdm(phrase_dict.items())
    for example, utterances in pbar:
        pbar.set_description(example)
        phrases[example] = {"female": [], "male": []}
        for gender in ["female", "male"]:
            for voice_info in all_us_voice_info[gender]:
                id = voice_info.name
                for short_long in ["short", "long"]:
                    anno = save_audio_anno(
                        example,
                        utterances[short_long],
                        voice_info,
                        short_long,
                        gender,
                        id,
                    )
                    phrases[example][gender].append(anno)

    write_json(phrases, join(savepath, "phrases.json"))
    return phrases


def _test_tts():
    import sounddevice as sd
    from datasets_turntaking.utils import load_waveform

    tts = TTSGoogleBeta()
    voice_info = tts.voices["female"][6]
    text = "Are you a student at this university?"
    pitch = -5
    filepath = "test.wav"
    words, times = tts.tts(text, filepath, voice_info=voice_info, pitch=pitch)

    y, sr = load_waveform(filepath)
    sd.play(y[0], samplerate=sr)


def _test_tts_normal():
    import sounddevice as sd
    from datasets_turntaking.utils import load_waveform

    tts = TTSGoogle()

    text = "Are you a student at this university?"
    voice_info = tts.voices["female"][5]
    filepath = "test_normal.wav"
    tts.tts(text, filepath, voice_info=voice_info)

    y, sr = load_waveform(filepath)
    sd.play(y[0], samplerate=sr)


def _test_process_path():
    from os.path import basename

    tts = TTSGoogle()

    path = "conv_ssl/evaluation/phrases.json"

    tts.process_short_long_phrases(path, savepath="assets/phrases")

    data = read_json(path)
    for example, utterances in data.items():
        print(example)
        print("Short: ", utterances["short"])
        print("Long: ", utterances["long"])
        print("-" * 40)

    filepath = dialog[0]["audio"]


if __name__ == "__main__":
    phrases = extract_phrases()
