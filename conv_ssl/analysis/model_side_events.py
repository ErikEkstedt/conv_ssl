import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets_turntaking import DialogAudioDM


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Evaluate model"""
    cfg_dict = OmegaConf.to_object(cfg)
    cfg_dict = dict(cfg_dict)

    # Load model
    model = VPModel.load_from_checkpoint(cfg.checkpoint_path, strict=False)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    # Load data
    print("Load Data")
    print("Num Workers: ", cfg.data.num_workers)
    print("Batch size: ", cfg.data.batch_size)
    print(cfg.data.datasets)
    dm = DialogAudioDM(
        datasets=cfg.data.datasets,
        type=cfg.data.type,
        audio_duration=cfg.data.audio_duration,
        audio_normalize=cfg.data.audio_normalize,
        audio_overlap=cfg.data.audio_overlap,
        sample_rate=cfg.data.sample_rate,
        vad_hz=model.frame_hz,
        vad_horizon=model.VAP.horizon,
        vad_history=cfg.data.vad_history,
        vad_history_times=cfg.data.vad_history_times,
        flip_channels=False,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    dm.prepare_data()
    dm.setup()

    # Extract Events
    # From dataset -> Turn-taking events
    # From model -> her the model "takes actions"

    # Extract model data

    # Save model data.
    #   keep only important data?

    # This shoould be in another script?
    # Visualize data centered on event
    # Spectrogram stereo visualization
    # Model output and top k probs
    # matplotlib? Web?


if __name__ == "__main__":

    model = load_model()
