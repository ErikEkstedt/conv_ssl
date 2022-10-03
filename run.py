from argparse import ArgumentParser
from conv_ssl.model import VPModel
from conv_ssl.utils import (
    everything_deterministic,
    get_tg_vad_list,
    read_json,
    write_json,
)
from conv_ssl.evaluation.duration import read_text_grid
import torch

everything_deterministic()


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--checkpoint", type=str, default="example/cpc_48_50hz_15gqq5s5.ckpt"
    )
    parser.add_argument(
        "-w",
        "--wav",
        default="example/student_long_female_en-US-Wavenet-G.wav",
    )
    parser.add_argument(
        "-v",
        "--voice_activity",
        type=str,
        default="example/vad_list.json",
    )
    parser.add_argument(
        "-tg",
        "--text_grid",
        type=str,
        default=None,
        # default="example/student_long_female_en-US-Wavenet-G.TextGrid",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="vap_output.json",
    )
    args = parser.parse_args()
    return args


def serialize_sample(sample):
    return {"vad": sample["vad"].tolist(), "waveform": sample["waveform"].tolist()}


if __name__ == "__main__":

    args = get_args()

    tg = None
    vad_list = None
    if args.voice_activity is not None:
        vad_list = read_json(args.voice_activity)

    if args.text_grid is not None:
        tg = read_text_grid(args.text_grid)
        vad_list = get_tg_vad_list(tg)

    print("Load Model: ", args.checkpoint)
    model = VPModel.load_from_checkpoint(args.checkpoint)
    model = model.eval()

    if model.mono:
        assert (tg is not None) or (vad_list is not None), "Mono model requires VAD"

    if torch.cuda.is_available():
        _ = model.to("cuda")

    print("Wavfile: ", args.wav)
    print("VA-list: ", args.voice_activity)
    print("TextGrid: ", args.text_grid)

    # get sample and process
    sample = model.load_sample(args.wav, vad_list)
    loss, out, probs, sample = model.output(sample)

    # Save
    data = {
        "loss": {"vp": loss["vp"].item(), "frames": loss["frames"].tolist()},
        "probs": out["logits_vp"].softmax(-1).tolist(),
        "labels": out["va_labels"].tolist(),
        "p": probs["p"].tolist(),
        "p_bc": probs["bc_prediction"].tolist(),
        "model": {
            "sample_rate": model.sample_rate,
            "frame_hz": model.frame_hz,
            "checkpoint": args.checkpoint,
        },
        "va": vad_list,
    }

    if tg is not None:
        data["words"] = tg["words"]
        data["phones"] = tg["phones"]

    write_json(data, args.output)
    print("Wrote output -> ", args.output)
