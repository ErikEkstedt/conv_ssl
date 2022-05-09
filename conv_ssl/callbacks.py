import torch
import pytorch_lightning as pl
import wandb

from conv_ssl.augmentations import flatten_pitch_batch


class SymmetricSpeakersCallback(pl.Callback):
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

    def on_train_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.get_symmetric_batch(batch)

    def on_test_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.get_symmetric_batch(batch)

    def on_val_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.get_symmetric_batch(batch)


class FlattenPitchCallback(pl.Callback):
    """ """

    def __init__(
        self,
        target_f0: int = -1,
        statistic: str = "mean",
        stats_frame_length: int = 800,
        stats_hop_length: int = 320,
        sample_rate: int = 16000,
        to_mono: bool = True,
    ):
        super().__init__()
        self.statistic = statistic
        self.stats_frame_length = stats_frame_length
        self.stats_hop_length = stats_hop_length
        self.target_f0 = target_f0
        self.sample_rate = sample_rate
        self.to_mono = to_mono

    def flatten_pitch(self, batch, device):
        """Appends a flipped version of the batch-samples"""
        flat_waveform = flatten_pitch_batch(
            waveform=batch["waveform"].cpu(),
            vad=batch["vad"],
            target_f0=self.target_f0,
            statistic=self.statistic,
            stats_frame_length=self.stats_frame_length,
            stats_hop_length=self.stats_hop_length,
            sample_rate=self.sample_rate,
            to_mono=self.to_mono,
        )
        batch["waveform"] = flat_waveform.to(device)
        return batch

    def on_test_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.flatten_pitch(batch, device=pl_module.device)

    def on_val_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.flatten_pitch(batch, device=pl_module.device)

    def on_train_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = self.flatten_pitch(batch, device=pl_module.device)


class WandbArtifactCallback(pl.Callback):
    def upload(self, trainer):
        run = trainer.logger.experiment
        print(f"Ending run: {run.id}")
        artifact = wandb.Artifact(f"{run.id}_model", type="model")
        for path, val_loss in trainer.checkpoint_callback.best_k_models.items():
            print(f"Adding artifact: {path}")
            artifact.add_file(path)
        run.log_artifact(artifact)

    def on_train_end(self, trainer, pl_module):
        print("Training End ---------------- Custom Upload")
        self.upload(trainer)

    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, KeyboardInterrupt):
            print("Keyboard Interruption ------- Custom Upload")
            self.upload(trainer)
