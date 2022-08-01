from datasets_turntaking.switchboard import load_switchboard
from datasets_turntaking.fisher import load_fisher

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        dset = load_switchboard(split=split)

    for split in ["train", "val", "test"]:
        dset = load_switchboard(split=split)
