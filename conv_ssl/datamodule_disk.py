from os.path import join
from glob import glob
from os import cpu_count

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from conv_ssl.utils import read_txt


class DiskDataset(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.root = root
        self.sample_paths = glob(join(root, "*.pt"))

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        return torch.load(self.sample_paths[idx])


class DiskDM(pl.LightningDataModule):
    def __init__(self, root, batch_size=4, num_workers=0):
        super().__init__()
        self.root = root
        self.train_path = join(root, "train")
        self.val_path = join(root, "val")
        self.test_path = join(root, "test")

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage="fit"):
        if stage == "test":
            self.test_dset = DiskDataset(self.test_path)
        else:
            self.train_dset = DiskDataset(self.train_path)
            self.val_dset = DiskDataset(self.val_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )


class DiskDatasetFiles(Dataset):
    def __init__(self, sample_paths) -> None:
        super().__init__()
        self.sample_paths = sample_paths

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        return torch.load(self.sample_paths[idx])


class DiskDMFiles(pl.LightningDataModule):
    def __init__(
        self,
        root,
        train_files=None,
        val_files=None,
        test_files=None,
        batch_size=4,
        num_workers=0,
    ):
        super().__init__()
        self.root = root
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files

        self.batch_size = batch_size
        self.num_workers = num_workers

    def init_paths(self, files):
        sample_paths = []

        sessions = read_txt(files)
        for session_number in sessions:
            tmp_paths = glob(join(self.root, f"{session_number}*.pt"))
            for p in tmp_paths:
                sample_paths.append(p)
        return sample_paths

    def setup(self, stage="fit"):
        if stage == "test":
            self.test_paths = self.init_paths(self.test_files)
            self.test_dset = DiskDatasetFiles(self.test_paths)
        else:
            self.train_paths = self.init_paths(self.train_files)
            self.val_paths = self.init_paths(self.val_files)

            self.train_dset = DiskDatasetFiles(self.train_paths)
            self.val_dset = DiskDatasetFiles(self.val_paths)

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        """argparse arguments for SoSIModel (based on yaml-config)"""
        parser = parent_parser.add_argument_group("DataModule from disk")
        parser.add_argument("--data_root", default="swb_dataset", type=str)
        parser.add_argument("--train_files", default=None, type=str)
        parser.add_argument("--val_files", default=None, type=str)
        parser.add_argument("--test_files", default=None, type=str)
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=cpu_count(), type=int)
        return parent_parser


if __name__ == "__main__":

    dm = DiskDM(root="swb_dataset")
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    batch = next(iter(dm.val_dataloader()))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    for batch in dm.train_dataloader():
        pass

    for batch in dm.val_dataloader():
        pass

    conf_root = "/home/erik/projects/conv_ssl/conv_ssl/config/swb_kfolds"
    train_files = join(conf_root, "1_fold_train.txt")
    val_files = join(conf_root, "1_fold_val.txt")
    dm = DiskDMFiles(train_files=train_files, val_files=val_files, root="swb_dataset")
    dm.setup()
    print("train: ", len(dm.val_dataloader()), len(dm.train_dataloader()))
    batch = next(iter(dm.val_dataloader()))

    conf_root = "/home/erik/projects/conv_ssl/conv_ssl/config/swb_kfolds"
    train_files = join(conf_root, "3_fold_train.txt")
    val_files = join(conf_root, "3_fold_val.txt")
    dm = DiskDMFiles(train_files=train_files, val_files=val_files, root="swb_dataset")
    dm.setup()
    print("train: ", len(dm.val_dataloader()), len(dm.train_dataloader()))
    batch = next(iter(dm.val_dataloader()))
