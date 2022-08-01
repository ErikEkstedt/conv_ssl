from os import makedirs
from os.path import join
import torch
from tqdm import tqdm

from conv_ssl.model import VPModel
from conv_ssl.utils import everything_deterministic
from datasets_turntaking import DialogAudioDM

everything_deterministic()


def video_data_single(
    model,
    dset,
    idx,
    audio_duration=10,
    audio_overlap=5,
    batch_size=8,
    savepath="assets/video",
):
    """
    Extract video data from single dialog (defined by `idx`)
    """

    # Extract all data for video
    d = dset.get_dialog_sample(idx)
    batches = dset.dialog_to_batch(
        d,
        audio_duration=audio_duration,
        audio_overlap=audio_overlap,
        batch_size=batch_size,
    )

    # Combine all data
    start_frame = int(audio_overlap * dm.vad_hz)
    start_sample = int(audio_overlap * dm.sample_rate)
    video_data = {"waveform": [], "va": [], "vh": [], "p": [], "p_bc": [], "logits": []}
    losses = []
    for i, batch in enumerate(tqdm(batches)):
        loss, out, probs, batch = model.output(batch)
        losses.append(loss["total"])
        tmp_batch_size = out["logits_vp"].shape[0]
        start_batch = 0
        if i == 0:
            video_data["waveform"].append(batch["waveform"][0].to("cpu"))
            video_data["va"].append(batch["vad"][0].to("cpu"))
            video_data["vh"].append(batch["vad_history"][0].to("cpu"))
            video_data["p"].append(probs["p"][0].to("cpu"))
            video_data["p_bc"].append(probs["bc_prediction"][0].to("cpu"))
            video_data["logits"].append(out["logits_vp"][0].to("cpu"))
            start_batch = 1
        for n in range(start_batch, tmp_batch_size):
            video_data["waveform"].append(batch["waveform"][n, start_sample:].to("cpu"))
            video_data["va"].append(batch["vad"][n, start_frame:].to("cpu"))
            video_data["vh"].append(batch["vad_history"][n, start_frame:].to("cpu"))
            video_data["p"].append(probs["p"][n, start_frame:].to("cpu"))
            video_data["p_bc"].append(probs["bc_prediction"][n, start_frame:].to("cpu"))
            video_data["logits"].append(out["logits_vp"][n, start_frame:].to("cpu"))

    for name, vallist in video_data.items():
        video_data[name] = torch.cat(vallist)

    # Add additional info
    video_data["loss"] = torch.stack(losses).mean()
    video_data["vap_bins"] = model.VAP.vap_bins.cpu()
    video_data["session"] = d["session"][0]

    # save to disk
    makedirs(savepath, exist_ok=True)

    filename = join(savepath, f"{d['session'][0]}_video_data.pt")
    torch.save(video_data, filename)
    print("Saved -> ", filename)

    return video_data


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--savepath", type=str)
    parser.add_argument("--n", type=int, default=3, help="Number of videos")
    parser.add_argument("--session", type=str, default=None, help="Specific session")

    args = parser.parse_args()

    # Load model
    model = VPModel.load_from_checkpoint(args.checkpoint, strict=False)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    # Load dialog
    # Load data
    data_conf = DialogAudioDM.load_config()
    DialogAudioDM.print_dm(data_conf)
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hz=model.frame_hz,
        vad_horizon=model.VAP.horizon,
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        flip_channels=False,
        batch_size=1,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup(None)

    if args.session is not None:
        idx = dm.test_dset.dataset["session"].index(args.session)
        video_data = video_data_single(
            model, dset=dm.test_dset, idx=idx, savepath=args.savepath
        )
    else:
        # select 20 first test-videos
        for idx in range(args.n):
            video_data = video_data_single(
                model, dset=dm.test_dset, idx=idx, savepath=args.savepath
            )
