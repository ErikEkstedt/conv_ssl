from os.path import join
import pytorch_lightning as pl
from tqdm import tqdm


from datasets_turntaking import DialogAudioDM


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


def add_animator_callback(args, callbacks):
    data_conf = DialogAudioDM.load_config(path=args.data_conf, args=args)
    val_hf_dataset = get_dialog_audio_datasets(
        datasets=data_conf["dataset"]["datasets"], split="val"
    )
    sample_dset = DialogIPU(
        dataset=val_hf_dataset,
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hop_time=data_conf["dataset"]["vad_hop_time"],
        vad_bin_sizes=data_conf["dataset"]["vad_bin_sizes"],
    )
    callbacks.append(
        AnimationCallback(
            sample_dset,
            start_epoch=args.animation_epoch_start,
            n_ani=args.animation_n,
        )
    )
    return callbacks


def ani_debug():
    from argparse import ArgumentParser
    from datasets_turntaking.dm_dialog_audio import (
        DialogAudioDM,
        DialogIPU,
        get_dialog_audio_datasets,
        print_dm,
    )

    parser = ArgumentParser()
    parser = DialogAudioDM.add_data_specific_args(parser)
    parser = ULMProjection.add_model_specific_args(parser)
    args = parser.parse_args()
    data_conf = DialogAudioDM.load_config(path=args.data_conf, args=args)
    # data_conf["dataset"]["type"] = "sliding"
    print_dm(data_conf, args)

    data_conf = DialogAudioDM.load_config(path=args.data_conf, args=args)
    val_hf_dataset = get_dialog_audio_datasets(
        datasets=data_conf["dataset"]["datasets"], split="val"
    )
    sample_dset = DialogIPU(
        dataset=val_hf_dataset,
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hop_time=data_conf["dataset"]["vad_hop_time"],
        vad_bin_sizes=data_conf["dataset"]["vad_bin_sizes"],
    )

    diter = iter(sample_dset)

    conf = ULMProjection.load_config(path=args.conf, args=args)
    model = ULMProjection(conf)

    batch = next(diter)

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    pass
