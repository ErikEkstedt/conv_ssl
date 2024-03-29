from os.path import basename, dirname, join, exists
import torch

from conv_ssl.model import VPModel
from datasets_turntaking import DialogAudioDM


def run_path_to_project_id(run_path):
    id = basename(run_path)  # 1xon133f
    project = dirname(run_path)  #  USER_NAME/PROJECT
    return project, id


def run_path_to_artifact_url(run_path, version="v0"):
    """
    run_path: "how_so/ULMProjection/1xon133f"

    artifact_url = "how_so/ULMProjection/model-1xon133f:v1"
    """
    project, id = run_path_to_project_id(run_path)

    artifact_path = project + "/" + "model-" + id + ":" + version
    return artifact_path


def get_checkpoint(run_path, version="v0", artifact_dir="./artifacts"):
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
    import wandb

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
    import wandb

    if not run_path.startswith("/"):
        run_path = "/" + run_path

    api = wandb.Api()
    run = api.run(run_path)
    return run


def load_model(checkpoint_path=None, run_path=None, eval=True, strict=True, **kwargs):
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint(run_path=run_path, **kwargs)
    model = VPModel.load_from_checkpoint(checkpoint_path, strict=strict)
    if torch.cuda.is_available():
        model = model.to("cuda")

    if eval:
        model = model.eval()
    return model


def load_dm(
    model=None,
    vad_hz=100,
    horizon=2,
    batch_size=4,
    num_workers=4,
    audio_duration=10,
    audio_overlap=1,
):
    data_conf = DialogAudioDM.load_config()

    if model is not None:
        horizon = round(sum(model.conf["vad_projection"]["bin_times"]), 2)
        vad_hz = model.frame_hz

    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        # audio_duration=data_conf["dataset"]["audio_duration"],
        audio_duration=audio_duration,
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=audio_overlap,
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hz=vad_hz,
        vad_horizon=horizon,
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        flip_channels=False,  # don't flip on evaluation
        batch_size=batch_size,
        num_workers=num_workers,
    )
    dm.prepare_data()
    dm.setup(None)
    return dm


# Temporary
def load_paper_versions(checkpoint_path, savepath=None):
    """
    The code was reformatted and simplified and so some paramter names were changed.

    This functions can load the checkpoints (at the paper version) and replace older names
    to create a new state_dict appropriate for the new version

    WARNING!
    The optimizer state is not changed so will probably be bad to continue training with that optimizer
    """

    print("Old Paper version checkpoint -> new")

    dir = dirname(checkpoint_path)
    name = basename(checkpoint_path)

    chpt = torch.load(checkpoint_path)
    sd = chpt["state_dict"]
    from_to = {
        "net.projection_head.weight": "net.vap_head.projection_head.weight",
        "net.projection_head.bias": "net.vap_head.projection_head.bias",
    }
    new_sd = {}
    for param, weight in sd.items():
        if param in from_to:
            print(param, "->", from_to[param])
            param = from_to[param]
        new_sd[param] = weight
    chpt["state_dict"] = new_sd
    if savepath is None:
        new_name = name.replace(".ckpt", "_new.ckpt")
        savepath = join(dir, new_name)
    torch.save(chpt, savepath)
    return savepath
