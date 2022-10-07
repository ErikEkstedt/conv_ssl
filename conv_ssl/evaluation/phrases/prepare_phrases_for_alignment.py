from argparse import ArgumentParser
from os.path import join, basename
from glob import glob
import string

from conv_ssl.utils import read_json


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="assets/phrases_beta")
    args = parser.parse_args()

    audio_path = join(args.data, "audio")
    anno_path = join(args.data, "annotation")

    wav_paths = glob(join(audio_path, "*.wav"))
    wav_paths.sort()

    for wav_path in wav_paths:
        name = basename(wav_path).replace(".wav", "")
        text = read_json(join(anno_path, name + ".json"))["text"]
        text = text.replace("-", " ")
        text = text.translate(str.maketrans("", "", string.punctuation)).lower()
        new_txt_path = join(audio_path, name + ".txt")
        with open(new_txt_path, "w") as text_file:
            text_file.write(text)
