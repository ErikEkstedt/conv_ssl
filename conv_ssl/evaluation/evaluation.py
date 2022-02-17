from os import makedirs, environ

import torch
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import WandbLogger

from conv_ssl.evaluation.utils import load_model, load_dm
from conv_ssl.utils import everything_deterministic

# from vad_turn_taking import DialogEvents, ProjectionCodebook
# import matplotlib.pyplot as plt
# from conv_ssl.plot_utils import plot_next_speaker_probs, plot_all_labels, plot_window
# import matplotlib as mpl
# mpl.use("TkAgg")


everything_deterministic()

# TODO: make the test such that it uploads to wandb (not lightning logs) to get all results in one place


class SymmetricSpeakersCallback(Callback):
    """
    This callback "flips" the speakers such that we get a fair evaluation not dependent on the
    biased speaker-order / speaker-activity

    The audio is mono which requires no change.

    The only change we apply is to flip the channels in the VAD-tensor and get the corresponding VAD-history
    which is defined as the ratio of speaker 0 (i.e. vad_history_flipped = 1 - vad_history)
    """

    def get_symmetric_batch(self, batch):
        """Appends a flipped version of the batch-samples"""
        for k, v in batch.items():
            if k == "vad":
                flipped = torch.stack((v[..., 1], v[..., 0]), dim=-1)
            elif k == "vad_history":
                flipped = 1.0 - v
            else:
                flipped = v
            if isinstance(v, torch.Tensor):
                batch[k] = torch.cat((v, flipped))
            else:
                batch[k] = v + flipped
        return batch

    def on_test_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.get_symmetric_batch(batch)

    def on_val_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.get_symmetric_batch(batch)


def test(model, dm, max_batches=None, project="VPModelTest"):
    savedir = "runs/" + project
    makedirs(savedir, exist_ok=True)
    logger = WandbLogger(
        save_dir=savedir,
        project=project,
        name=model.run_name,
        log_model=False,
    )

    if max_batches is not None:
        trainer = Trainer(
            gpus=-1,
            limit_test_batches=max_batches,
            deterministic=True,
            logger=logger,
            callbacks=[SymmetricSpeakersCallback()],
        )
    else:
        trainer = Trainer(
            gpus=-1,
            deterministic=True,
            logger=logger,
            callbacks=[SymmetricSpeakersCallback()],
        )
    result = trainer.test(model, dataloaders=dm.test_dataloader(), verbose=False)
    return result


if __name__ == "__main__":

    # run_path: found in wandb information tab
    # run_path = "how_so/VPModel/2wbyll6r"  # discrete
    run_path = "how_so/VPModel/27ly86w3"  # independent (same bin size)
    # run_path = "how_so/VPModel/2y480ysa"  # comparative broken (no artifact)
    model = load_model(run_path=run_path)
    dm = load_dm(model, batch_size=4, num_workers=4)
    result = test(model, dm)
    # batch = next(iter(dm.val_dataloader()))
