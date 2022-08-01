from argparse import ArgumentParser
import torch
import matplotlib.pyplot as plt

from conv_ssl.evaluation.evaluation_phrases import plot_sample
from conv_ssl.utils import read_json, load_waveform


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-w",
        "--wav",
        type=str,
        default="example/student_long_female_en-US-Wavenet-G.wav",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="vap_output.json",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    data = read_json(args.file)

    sample = {
        "waveform": load_waveform(
            args.wav,
            sample_rate=data["model"]["sample_rate"],
            normalize=True,
            mono=True,
        )[0],
        "vad": data["va"],
    }

    if "phones" in data:
        sample["phones"] = data["phones"]

    if "words" in data:
        sample["words"] = data["words"]

    prob_next_speaker = torch.tensor(data["p"])[0, :, 0]

    fig, ax = plot_sample(
        prob_next_speaker,
        sample=sample,
        sample_rate=data["model"]["sample_rate"],
        frame_hz=data["model"]["frame_hz"],
    )
    plt.show()
