from os.path import join
from tqdm import tqdm
import pytorch_lightning as pl
import wandb


class AnimationCallback(pl.Callback):
    def __init__(
        self,
        sample_dset,
        n_ani=-1,
        frame_step=5,
        start_epoch=1,
        cache_path="/tmp/vad_animation",
    ):
        super().__init__()
        self.n_ani = n_ani
        self.sample_dset = sample_dset
        self.start_epoch = start_epoch
        self.cache_path = cache_path
        self.frame_step = frame_step

    def create_animations(self, model):
        paths = []
        for i, d in tqdm(
            enumerate(self.sample_dset), desc="Animation", total=self.n_ani
        ):
            if self.n_ani > 0 and i == self.n_ani:
                break

            path = join(self.cache_path, f"ani_{i}.mp4")
            model.animate_sample(
                input_ids=d["q"],
                waveform=d["waveform"],
                vad=d["vad"],
                frame_step=self.frame_step,
                path=path,
            )
            paths.append(path)
        return paths

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if trainer.current_epoch < self.start_epoch:
            return None

        paths = self.create_animations(model=pl_module)
        for i, p in enumerate(paths):
            wandb.log(
                {
                    f"animation_{i}": wandb.Video(
                        data_or_path=paths[i], fps=10, format="mp4"
                    )
                }
            )
        return None
