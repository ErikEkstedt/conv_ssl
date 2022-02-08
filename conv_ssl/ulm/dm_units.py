from argparse import ArgumentParser
from glob import glob
from os.path import join
from os import cpu_count
from typing import Optional

import torch
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class SegmentDataset(Dataset):
    def __init__(self, root, k=100, split="train"):
        super().__init__()
        self.k = k
        self.root = root  # data/kmeans_files/hubert_base
        self.split = split
        self.folder = join(self.root, split)  # data/kmeans_files/hubert_base/train
        self.files = glob(join(self.folder, f"*_{k}_*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return torch.load(self.files[index])  # {'q', 'vad_label'}


class SegmentDM(pl.LightningDataModule):
    def __init__(self, root, batch_size=4, pin_memory=True, num_workers=4):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if stage == "all":
            self.train_dset = SegmentDataset(self.root, split="train")
            self.val_dset = SegmentDataset(self.root, split="val")
            self.test_dset = SegmentDataset(self.root, split="test")
        elif stage == "test":
            self.test_dset = SegmentDataset(self.root, split="test")
        else:  # if stage == "fit" or stage is None:
            self.train_dset = SegmentDataset(self.root, split="train")
            self.val_dset = SegmentDataset(self.root, split="val")

    def collate_fn(self, batch):
        segments = []
        waveforms = []
        vads = []
        vad_labels = []
        vad_history = []
        lengths = []
        for seg in batch:
            segments.append(seg["q"])
            lengths.append(len(seg["q"]))

            vad_labels.append(seg["vad_label"])
            if "waveform" in seg:
                waveforms.append(seg["waveform"])

            if "vad" in seg:
                vads.append(seg["vad"])

            if "vad_history" in seg:
                vad_history.append(seg["vad_history"])

        ret = {
            "q": pad_sequence(segments, batch_first=True),
            "vad_label": pad_sequence(vad_labels, batch_first=True),
            "lengths": torch.tensor(lengths),
        }
        if len(vads) > 0:
            ret["vad"] = pad_sequence(vads, batch_first=True)

        if len(vad_history) > 0:
            ret["vad_history"] = pad_sequence(vad_history, batch_first=True)

        if len(waveforms) > 0:
            ret["waveform"] = pad_sequence(waveforms, batch_first=True)
        return ret

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SegmentDM")
        n_cpus = cpu_count()
        parser.add_argument("--data_root", type=str)
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=n_cpus, type=int)
        parser.add_argument("--pin_memory", default=True, type=bool)
        return parent_parser


def test_animation(
    sample_path="tmp_data/kmeans_files/hubert_base/val_audio/k_100_120.pt",
):
    from conv_ssl.vad_pred_animation import VadPredAnimator
    from datasets_turntaking.features.vad import VadProjection
    import matplotlib as mpl

    mpl.use("tkagg")

    d = torch.load(sample_path)
    vad_labels = d["vad_label"]
    print("vad_labels: ", tuple(vad_labels.shape), vad_labels.dtype)
    codebook_vad = VadProjection(n_bins=8)
    vad_label_oh = codebook_vad(vad_labels)
    print("vad_label_oh: ", tuple(vad_label_oh.shape), vad_label_oh.dtype)

    vp = VadPredAnimator(
        waveform=d["waveform"],
        vad=d["vad"],
        vad_label_oh=vad_label_oh.view(vad_label_oh.shape[0], 2, 4),
        # bin_sizes=[20, 40, 60, 80],
        bin_sizes=[10, 20, 30, 40],
    )
    vp.animation()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = SegmentDM.add_data_specific_args(parser)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 50)

    dm = SegmentDM(root="dataset_units/hubert_base")
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    for k, v in batch.items():
        if k == "lengths":
            print(f"{k}: {v}")
        elif isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    dset = SegmentDataset(root="tmp_data/kmeans_files/hubert_base", split="val_audio")
    print("audio dset: ", len(dset))

    test_animation()
