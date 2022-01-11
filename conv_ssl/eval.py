from os.path import join, exists, basename, dirname
import torch
import wandb

import pytorch_lightning as pl

from conv_ssl.ulm_projection import ULMProjection
from conv_ssl.models import ProjectionMetricCallback


def run_path_to_project_id(run_path):
    id = basename(run_path)  # 1xon133f
    project = dirname(run_path)  #  how_so/ULMProjection
    return project, id


def run_path_to_artifact_url(run_path, version="v1"):
    """
    run_path: "how_so/ULMProjection/1xon133f"

    artifact_url = "how_so/ULMProjection/model-1xon133f:v1"
    """
    project, id = run_path_to_project_id(run_path)

    artifact_path = project + "/" + "model-" + id + ":" + version
    return artifact_path


def get_checkpoint(run_path, version="v1", artifact_dir="./artifacts"):
    """
    On information tab in WandB find 'Run Path' and copy to clipboard

    ---------------------------------------------------------
    run_path:       how_so/ULMProjection/1tokrds0
    ---------------------------------------------------------
    project:        how_so/ULMProjection
    id:             1tokrds0
    artifact_url:   how_so/ULMProjection/model-1xon133f:v1
    checkpoint:     ${artifact_dir}/model-3hysqnmt:v1/model.ckpt
    ---------------------------------------------------------
    """
    # project, id = run_path_to_project_id(run_path)
    artifact_url = run_path_to_artifact_url(run_path, version)
    model_name = basename(artifact_url)
    checkpoint = join(artifact_dir, model_name, "model.ckpt")

    if not exists(checkpoint):
        # URL: always '/'
        with wandb.init() as run:
            artifact = run.use_artifact(artifact_url, type="model")
            _ = artifact.download()
    return checkpoint


def load_metadata(run_path):
    if not run_path.startswith("/"):
        run_path = "/" + run_path

    api = wandb.Api()
    run = api.run(run_path)
    return run


def load_dm(split="val", batch_size=10, vad_history=True):
    from datasets_turntaking.dm_dialog_audio import DialogAudioDM, print_dm
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = DialogAudioDM.add_data_specific_args(parser)
    args = parser.parse_args()

    ##############
    # DataModule #
    ##############
    data_conf = DialogAudioDM.load_config(path=args.data_conf, args=args)
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        audio_include_ratio=data_conf["dataset"]["audio_include_ratio"],
        audio_context_duration=data_conf["dataset"]["audio_context_duration"],
        ipu_min_time=data_conf["dataset"]["ipu_min_time"],
        ipu_pause_time=data_conf["dataset"]["ipu_pause_time"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hop_time=data_conf["dataset"]["vad_hop_time"],
        vad_bin_sizes=data_conf["dataset"]["vad_bin_sizes"],
        vad_history=vad_history,
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=batch_size,
        num_workers=args.num_workers,
    )
    dm.prepare_data()
    print_dm(data_conf, args)

    if split == "test":
        dm.setup("test")
        return dm.test_dataloader()
    else:
        dm.setup("fit")
        return dm.val_dataloader()


def eval(model, dloader):
    trainer = pl.Trainer(callbacks=[ProjectionMetricCallback()], gpus=-1)
    return trainer.test(model, dataloaders=dloader)


# iterate over test-set
# find N highest/lowest loss samples
# -> create animation of those


if __name__ == "__main__":

    run_path = "how_so/ULMProjection_delete/20qbod08"
    checkpoint_path = get_checkpoint(run_path, version="v0")
    # checkpoint = torch.load(checkpoint_path)
    # conf = checkpoint["hyper_parameters"]["conf"]
    metadata = load_metadata(run_path)
    model = ULMProjection.load_from_checkpoint(checkpoint_path)
    dloader = load_dm()

    # Evaluate over entire dataloader
    out_test = eval(model, dloader)
    for k, v in out_test[0].items():
        print(f"{k}: {v}")

    # animation
    batch = next(iter(dloader))
    loss, out, batch, batch_size = model.shared_step(batch)

    # Requires FFMPEG: `conda install -c conda-forge ffmpeg`
    model.animate_sample(
        waveform=batch["waveform"][0],
        vad=batch["vad"][0],
        frame_step=5,  # 50 hz
        path="ulm_projection_vid.mp4",
    )
