from argparse import ArgumentParser
from os import cpu_count, makedirs
from os.path import join
from tqdm import tqdm

import torch
from datasets_turntaking import DialogAudioDM

from conv_ssl.utils import write_json


def save_samples(dm, root, max_batches=-1):
    makedirs(root, exist_ok=True)
    file_map = {}
    for dloader in [dm.val_dataloader(), dm.train_dataloader()]:
        for ii, batch in enumerate(dloader):
            if max_batches > 0 and ii == max_batches:
                break
            batch_size = batch["waveform"].shape[0]
            for i in range(batch_size):
                session = batch["session"][i]
                if session not in file_map:
                    file_map[session] = -1
                file_map[session] += 1
                n = file_map[session]
                sample = {
                    "waveform": batch["waveform"][i],
                    "vad": batch["vad"][i],
                    "vad_history": batch["vad_history"][i],
                    "dset_name": batch["dset_name"][i],
                    "session": batch["session"][i],
                }
                name = f"{session}_{n}.pt"
                torch.save(sample, join(root, name))
    write_json(file_map, join(root, "file_map.json"))


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
    DialogAudioDM.print_dm(data_conf, args)
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hz=100,
        vad_horizon=2,
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.prepare_data()
    dm.setup()

    save_samples(dm, args.dirpath, max_batches=args.max_batches)
