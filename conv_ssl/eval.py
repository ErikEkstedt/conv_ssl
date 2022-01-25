from os.path import join, exists, basename, dirname
import torch
import wandb
from tqdm import tqdm

import pytorch_lightning as pl

from conv_ssl.ulm_projection import ULMProjection
from conv_ssl.models import ProjectionMetricCallback, ProjectionMetrics


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


def trainer_eval(model, dloader):
    trainer = pl.Trainer(callbacks=[ProjectionMetricCallback()], gpus=-1)
    return trainer.test(model, dataloaders=dloader)


@torch.no_grad()
def manual_eval(model, dloader, n_high=10, n_low=5, max_iter=-1):
    def to_device(batch, device="cuda"):
        new_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                new_batch[k] = v.to(device)
            else:
                new_batch[k] = v
        return new_batch

    def get_sample_from_dict(i, batch):
        sample = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                sample[k] = v[i].unsqueeze(0).cpu()  # add batch dim
            elif isinstance(v, list):
                sample[k] = [v[i]]
            else:
                sample[k] = v[i]
        return sample

    assert n_high > 0
    assert n_low > 0

    model.eval()

    metrics = ProjectionMetrics()
    avg_loss = {"vp": [], "ar": [], "total": []}
    losses = {
        "low": torch.ones(n_low, device=model.device) * 9999,
        "high": torch.ones(n_high, device=model.device) * -9999,
    }
    outs = {"low": [None] * n_low, "high": [None] * n_high}
    samples = {"low": [None] * n_low, "high": [None] * n_high}

    metrics.reset()

    if max_iter > 0:
        pbar = tqdm(enumerate(dloader), total=max_iter)
    else:
        pbar = enumerate(dloader)

    for n_batch, batch in pbar:
        batch = to_device(batch, device=model.device)
        loss, out, batch, _ = model.shared_step(batch, reduction="none")

        # Metrics Update
        metrics.update(
            logits=out["logits_vp"], vad=batch["vad"], vad_label=batch["vad_label"]
        )

        batch_loss = loss["vp"].mean(dim=-1)  # (B,)
        avg_loss["total"].append(loss["total"].mean().cpu())
        avg_loss["vp"].append(loss["vp"].mean().cpu())
        if "ar" in loss:
            avg_loss["ar"].append(loss["ar"].mean().cpu())

        for i, tmp_l in enumerate(batch_loss):
            # Update low losses
            less = torch.where(tmp_l < losses["low"])[0]
            if len(less) > 0:
                update = losses["low"][less].argmax()  # update the highest
                losses["low"][update] = tmp_l
                samples["low"][update] = get_sample_from_dict(i, batch)
                outs["low"][update] = get_sample_from_dict(i, out)

            # Update high losses
            greater = torch.where(tmp_l > losses["high"])[0]
            if len(greater) > 0:
                update = losses["high"][greater].argmin()  # update the lowest
                losses["high"][update] = tmp_l
                samples["high"][update] = get_sample_from_dict(i, batch)
                outs["high"][update] = get_sample_from_dict(i, out)

        if max_iter > 0 and n_batch >= max_iter:
            break

    for k, v in avg_loss.items():
        if len(v) > 0:
            avg_loss[k] = torch.stack(v).mean()

    result = metrics.compute()
    return {
        "outs": outs,
        "samples": samples,
        "losses": losses,
        "avg_loss": avg_loss,
        "result": result,
    }


if __name__ == "__main__":
    run_path = "how_so/ULMProjection/1er9zvt6"
    checkpoint_path = get_checkpoint(run_path, version="v0")
    checkpoint = torch.load(checkpoint_path)
    conf = checkpoint["hyper_parameters"]["conf"]
    metadata = load_metadata(run_path)
    model = ULMProjection.load_from_checkpoint(checkpoint_path)
    dloader = load_dm(batch_size=4)
    if torch.cuda.is_available():
        model = model.to("cuda")

    # Evaluate over entire dataloader
    # out_test = trainer_eval(model, dloader)
    result = manual_eval(model, dloader, n_low=2, n_high=6, max_iter=100)
    for k, v in result["result"].items():
        print(f"{k}: {v}")

    # # Requires FFMPEG: `conda install -c conda-forge ffmpeg`
    # # animation
    # batch = next(iter(dloader))
    # loss, out, batch, batch_size = model.shared_step(batch, reduction="none")
    model.animate_sample(
        waveform=result["samples"]["high"][0]["waveform"][0],
        vad=result["samples"]["high"][0]["vad"][0],
        frame_step=5,  # 50 hz
        path="ulm_projection_vid.mp4",
    )
