from argparse import ArgumentParser
from os import cpu_count, makedirs
from os.path import join
from tqdm import tqdm

import torch
from datasets_turntaking import DialogAudioDM


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--dirpath", default="swb_dataset", type=str)
    parser.add_argument("--hz", default=100, type=int)
    parser.add_argument("--duration", default=10, type=float)
    parser.add_argument("--sample_rate", default=16000, type=int)
    parser.add_argument("--horizon", default=3, type=float)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--num_workers", default=cpu_count(), type=int)
    parser.add_argument("--max_batches", default=-1, type=int)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    data_conf = DialogAudioDM.load_config()
    data_conf["dataset"]["vad_hz"] = args.hz
    data_conf["dataset"]["vad_bin_times"] = [0.2, 0.4, 0.6, 0.8]
    DialogAudioDM.print_dm(data_conf, args)
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hz=data_conf["dataset"]["vad_hz"],
        vad_bin_times=data_conf["dataset"]["vad_bin_times"],
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.prepare_data()
    dm.setup()

    makedirs(args.dirpath, exist_ok=True)
    makedirs(join(args.dirpath, "train"), exist_ok=True)

    if args.max_batches > 0:
        pbar = tqdm(dm.train_dataloader(), total=args.max_batches)
    else:
        pbar = tqdm(dm.train_dataloader())
    n = 0
    for n_batch, batch in enumerate(pbar):
        batch_size = batch["waveform"].shape[0]
        if args.max_batches > 0 and n_batch >= args.max_batches:
            break
        for i in range(batch_size):
            sample = {
                "waveform": batch["waveform"][i],
                "vad": batch["vad"][i],
                "vad_history": batch["vad_history"][i],
                "dset_name": batch["dset_name"][i],
                "session": batch["session"][i],
            }
            torch.save(sample, join(args.dirpath, "train", f"d_{str(n).zfill(6)}.pt"))
            n += 1

    ################################################################
    # Validation
    split = "val"
    makedirs(join(args.dirpath, split), exist_ok=True)

    if args.max_batches > 0:
        pbar = tqdm(dm.val_dataloader(), total=args.max_batches)
    else:
        pbar = tqdm(dm.val_dataloader())
    n = 0
    for n_batch, batch in enumerate(pbar):
        batch_size = batch["waveform"].shape[0]
        if args.max_batches > 0 and n_batch >= args.max_batches:
            break
        for i in range(batch_size):
            sample = {
                "waveform": batch["waveform"][i],
                "vad": batch["vad"][i],
                "vad_history": batch["vad_history"][i],
                "dset_name": batch["dset_name"][i],
                "session": batch["session"][i],
            }
            torch.save(sample, join(args.dirpath, split, f"d_{str(n).zfill(6)}.pt"))
            n += 1
