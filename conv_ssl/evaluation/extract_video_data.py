import torch
from tqdm import tqdm
from conv_ssl.model import VPModel
from conv_ssl.utils import everything_deterministic
from datasets_turntaking import DialogAudioDM

everything_deterministic()


if __name__ == "__main__":

    # Load model
    # model_id = "120k8fdv"
    # checkpoint_path = get_checkpoint(run_path=model_id)
    # checkpoint_path = load_paper_versions(checkpoint_path)
    checkpoint_path = "./artifacts/model-120k8fdv:v0/model_new.ckpt"

    model = VPModel.load_from_checkpoint(checkpoint_path, strict=False)
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

    dset = dm.val_dset
    # chunk dialog -> segments -> order in batch (to fit gpu)
    audio_duration = 10
    audio_overlap = 5
    batch_size = 8
    # Extract all data for video
    d = dset.get_dialog_sample(0)
    batches = dset.dialog_to_batch(
        d,
        audio_duration=audio_duration,
        audio_overlap=audio_overlap,
        batch_size=batch_size,
    )
    # Combine all data
    start_frame = int(audio_overlap * dm.vad_hz)
    start_sample = int(audio_overlap * dm.sample_rate)
    video = {"waveform": [], "va": [], "vh": [], "p": [], "p_bc": [], "logits": []}
    for i, batch in enumerate(tqdm(batches)):
        loss, out, probs, batch = model.output(batch)
        tmp_batch_size = out["logits_vp"].shape[0]
        start_batch = 0
        if i == 0:
            video["waveform"].append(batch["waveform"][0].to("cpu"))
            video["va"].append(batch["vad"][0].to("cpu"))
            video["vh"].append(batch["vad_history"][0].to("cpu"))
            video["p"].append(probs["p"][0].to("cpu"))
            video["p_bc"].append(probs["bc_prediction"][0].to("cpu"))
            video["logits"].append(out["logits_vp"][0].to("cpu"))
            start_batch = 1
        for n in range(start_batch, tmp_batch_size):
            video["waveform"].append(batch["waveform"][n, start_sample:].to("cpu"))
            video["va"].append(batch["vad"][n, start_frame:].to("cpu"))
            video["vh"].append(batch["vad_history"][n, start_frame:].to("cpu"))
            video["p"].append(probs["p"][n, start_frame:].to("cpu"))
            video["p_bc"].append(probs["bc_prediction"][n, start_frame:].to("cpu"))
            video["logits"].append(out["logits_vp"][n, start_frame:].to("cpu"))

    for name, vallist in video.items():
        video[name] = torch.cat(vallist)
    video["vap_bins"] = model.VAP.vap_bins.cpu()

    # save to disk
    torch.save(video, "assets/video.pt")
