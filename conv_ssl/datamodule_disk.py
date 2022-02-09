from os.path import join
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


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
