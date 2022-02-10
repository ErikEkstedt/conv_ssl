import math
from os.path import basename, dirname, exists, join
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer, Callback

from conv_ssl.model import VPModel
from conv_ssl.plot_utils import plot_next_speaker_probs, plot_all_labels, plot_window
from conv_ssl.utils import everything_deterministic
from datasets_turntaking import DialogAudioDM

from vad_turn_taking import DialogEvents, ProjectionCodebook

import matplotlib as mpl

mpl.use("TkAgg")


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


def to_device(batch, device="cuda"):
    new_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            new_batch[k] = v.to(device)
        else:
            new_batch[k] = v
    return new_batch


def gaussian_kernel_1d_unidirectional(N, sigma=3):
    """

    N: points to include in smoothing (including current) i.e. N=5 => [t-4, t-3, t-2, t-1, 5]

    source:
        https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351
    """
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    kernel_size = (N - 1) * 2 + 1
    x_cord = torch.arange(kernel_size).unsqueeze(0)
    mean = (kernel_size - 1) / 2.0
    variance = sigma ** 2.0

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    scale = 1.0 / (2.0 * math.pi * variance)
    gaussian_kernel = scale * torch.exp(
        -torch.sum((x_cord - mean) ** 2.0, dim=0) / (2 * variance)
    )

    # only care about left half
    gaussian_kernel = gaussian_kernel[:N]

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return gaussian_kernel


def smooth_gaussian(x, N, sigma=1):
    kernel = gaussian_kernel_1d_unidirectional(N, sigma)

    # pad left
    xx = F.pad(x, pad=(N - 1, 0), mode="replicate")
    if xx.ndim == 2:
        xx = xx.unsqueeze(1)
    a = F.conv1d(xx, weight=kernel.view(1, 1, -1), bias=None, stride=1)
    return a.squeeze(1)


def explanation():
    vad_projection = ProjectionCodebook()
    print("on_silent_shift: ", tuple(vad_projection.on_silent_shift.shape))
    print("on_silent_hold: ", tuple(vad_projection.on_silent_hold.shape))
    print("on_active_shift: ", tuple(vad_projection.on_active_shift.shape))
    print("on_active_hold: ", tuple(vad_projection.on_active_hold.shape))
    print("--------------------------------------------")
    plot_all_labels(vad_projection, next_speaker=0)


def load_dm(model, test=False, vad_history=None, batch_size=4, num_workers=4):
    data_conf = DialogAudioDM.load_config()

    if vad_history is not None:
        data_conf["dataset"]["vad_history"] = vad_history

    # dm = DialogAudioDM(
    #     datasets=data_conf["dataset"]["datasets"],
    #     type=data_conf["dataset"]["type"],
    #     audio_duration=data_conf["dataset"]["audio_duration"],
    #     audio_normalize=data_conf["dataset"]["audio_normalize"],
    #     audio_overlap=data_conf["dataset"]["audio_overlap"],
    #     sample_rate=data_conf["dataset"]["sample_rate"],
    #     vad_hz=model.frame_hz,
    #     vad_bin_times=model.conf["vad_class_prediction"]["bin_times"],
    #     vad_history=data_conf["dataset"]["vad_history"],
    #     vad_history_times=data_conf["dataset"]["vad_history_times"],
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    # )

    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hz=model.frame_hz,
        vad_bin_times=model.conf["vad_projection"]["bin_times"],
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=batch_size,
        num_workers=num_workers,
    )
    dm.prepare_data()
    stage = "fit"
    if test:
        stage = "test"
    dm.setup(stage)
    return dm


def plot_batch(model, batch):
    """Plot a batch"""

    B = batch["vad"].shape[0]

    # Single
    plt.close("all")
    with torch.no_grad():
        batch = to_device(batch, model.device)
        loss, out, batch = model.shared_step(batch, reduction="none")
        probs = out["logits_vp"].softmax(dim=-1)
        vad = batch["vad"]
        p_next = model.vad_projection.get_next_speaker_probs(probs, vad).cpu()
        p_shift = model.vad_projection.speaker_prob_to_shift(p_next, vad)
        # TEST PLACES
        # Valid shift/hold
        valid = DialogEvents.find_valid_silences(
            batch["vad"],
            horizon=model.vad_projection.event_horizon,
            min_context=model.vad_projection.event_min_context,
            min_duration=model.vad_projection.event_min_duration,
            start_pad=model.vad_projection.event_start_pad,
            target_frames=model.vad_projection.event_target_duration,
        )
        hold, shift = DialogEvents.find_hold_shifts(batch["vad"])
        hold, shift = torch.logical_and(hold, valid.unsqueeze(-1)), torch.logical_and(
            shift, valid.unsqueeze(-1)
        )
    for b in range(B):
        _ = plot_next_speaker_probs(
            p_next[b].cpu(),
            shift_prob=p_shift[b].cpu(),
            vad=vad[b].cpu(),
            shift=shift[b].sum(dim=-1).cpu(),
            hold=hold[b].sum(dim=-1).cpu(),
        )


def evaluation(
    model, dloader, event_start_pad=None, event_target_duration=None, max_batches=None
):
    """
    Args:
        model:                  pl.LightningModule
        dloader:                DataLoader (torch)
        event_start_pad:        int, number of frames (silence after activity) until target
        event_target_duration:  int, number of frames of target (spanning this many frames)
        max_batches:            int, maximum number of batches to evaluate

    Return:
        result:                 Dict, containing keys: ['test/{metrics}']
    """
    old_pad = model.vad_projection.event_start_pad
    if event_start_pad is not None:
        model.vad_projection.event_start_pad = event_start_pad

    old_t_dur = model.vad_projection.event_target_duration
    if event_target_duration is not None:
        model.vad_projection.event_target_duration = event_target_duration

    # make sure the valid targets include the (changed?) target params
    pad = model.vad_projection.event_start_pad
    t_dur = model.vad_projection.event_target_duration
    old_min_duration = None
    if model.vad_projection.event_min_duration < (pad + t_dur):
        old_min_duration = model.vad_projection.event_min_duration
        model.vad_projection.event_min_duration = pad + t_dur

    # Actual Evaluation
    if max_batches is None:
        trainer = Trainer(gpus=-1)
    else:
        trainer = Trainer(gpus=-1, limit_test_batches=max_batches)

    result = trainer.test(model, dataloaders=dloader, verbose=False)
    result = result[0]

    # Restore model.vad_projection values to original
    if event_start_pad is not None:
        model.vad_projection.event_start_pad = old_pad

    if event_target_duration is not None:
        model.vad_projection.event_target_duration = old_t_dur

    if old_min_duration is not None:
        model.vad_projection.event_min_duration = old_min_duration
    return result


def metric_aggregation(
    model, dloader, pads=[10], durs=[10, 20, 40, 60], max_batches=None
):
    aggregate_results = {}
    for event_start_pad in pads:
        # print("event_start_pad: ", event_start_pad)
        pad_name = f"pad_{event_start_pad}"
        aggregate_results[pad_name] = {}
        for event_target_duration in durs:
            # print("event_target_duration: ", event_target_duration)
            tmp_res = evaluation(
                model,
                dloader,
                event_start_pad=event_start_pad,
                event_target_duration=event_target_duration,
                max_batches=max_batches,
            )
            aggregate_results[pad_name][f"dur_{event_target_duration}"] = tmp_res

            print(
                f'p: {event_start_pad}, d: {event_target_duration}, F1: {tmp_res["test/f1_weighted"]}'
            )
    return aggregate_results


class AblationCallback(Callback):
    def __init__(self, vad_history=None, vad=None) -> None:
        super().__init__()

        if vad_history is None and vad is None:
            raise NotImplementedError(
                "Must provide a valid ablation, NOOP not implemented"
            )

        assert vad_history in ["reverse", "equal", None], "vad_history error"
        self.vad_history = vad_history
        self.vad = vad

    def on_test_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        if self.vad_history is not None:
            if "vad_history" in batch:
                if self.vad_history == "reverse":
                    batch["vad_history"] = 1 - batch["vad_history"]
                else:
                    batch["vad_history"] = torch.ones_like(batch["vad_history"]) * 0.5

        if self.vad in ["reverse", "zero"]:
            if self.vad == "reverse":
                batch["vad"] = torch.stack(
                    (batch["vad"][..., 1], batch["vad"][..., 0]), dim=-1
                )
            else:
                batch["vad"] = torch.zeros_like(batch["vad"])


def test_causality(batch, model):
    step = 500

    batch = to_device(batch, model.device)
    batch["waveform"].requires_grad = True
    loss, _, _ = model.shared_step(batch, reduction="none")
    loss["vp"][:, step].norm().backward()
    g = batch["waveform"].grad.abs().cpu()

    b = 0
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(model.run_name)
    ax.plot(g[b] / g[b].max())
    plt.show()


if __name__ == "__main__":
    everything_deterministic()

    # TODO: load artifact from wandb

    checkpoints = [
        # "checkpoints/cpc/cpc_04_14.ckpt",
        # "checkpoints/cpc/cpc_04_24.ckpt",
        # "checkpoints/cpc/cpc_04_44.ckpt",
        "checkpoints/cpc/cpc_04_14_reg.ckpt",
    ]
    # chpt = "checkpoints/cpc/cpc_04_14.ckpt"
    # chpt = "checkpoints/cpc/cpc_04_24.ckpt"

    results = {}
    for chpt in checkpoints:
        model = VPModel.load_from_checkpoint(chpt)
        model = model.to("cuda")
        dm = load_dm(model, test=True)
        # trainer = Trainer(gpus=-1, limit_test_batches=20, deterministic=True)
        trainer = Trainer(gpus=-1, deterministic=True)
        result = trainer.test(model, dataloaders=dm.test_dataloader(), verbose=False)
        name = basename(chpt)
        results[name] = result[0]

    torch.save(results, "evaluation_reg_scores.pt")
    # torch.save(results, "evaluation_scores.pt")
    # r = torch.load("evaluation_scores.pt")

    r = torch.load("evaluation_reg_scores.pt")

    # #######################################################
    # chpt = "checkpoints/cpc/cpc_04_14_reg.ckpt"
    # model = VPModel.load_from_checkpoint(chpt)
    # model = model.to("cuda")
    # model.eval()
    # dm = load_dm(model, test=False)
    # diter = iter(dm.val_dataloader())
    #
    # trainer = Trainer(gpus=-1, limit_test_batches=50, deterministic=True)
    # result = trainer.test(model, dataloaders=dm.val_dataloader(), verbose=False)
    # print(result[0])

    # print(model.val_metric)
    # batch = next(diter)
    # batch = to_device(batch, model.device)
    # model.val_metric.reset()
    # with torch.no_grad():
    #     loss, out, batch = model.shared_step(batch)
    #     next_probs, pre_probs = model.get_next_speaker_probs(
    #         out["logits_vp"], vad=batch["vad"]
    #     )
    #     events = model.val_metric.extract_events(batch["vad"])
    #     model.val_metric.update(
    #         next_probs, vad=batch["vad"], events=events, bc_pre_probs=pre_probs
    #     )
    #     r = model.val_metric.compute()
    #
    # b = 0
    # fig, ax = plot_window(
    #     next_probs[b].cpu(),
    #     vad=batch["vad"][b].cpu(),
    #     hold=events["hold"][b].cpu(),
    #     shift=events["shift"][b].cpu(),
    #     pre_hold=events["pre_hold"][b].cpu(),
    #     pre_shift=events["pre_shift"][b].cpu(),
    #     backchannels=events["backchannels"][b].cpu(),
    #     plot_kwargs=dict(
    #         alpha_event=0.2,
    #         alpha_vad=0.6,
    #         shift_hatch=".",
    #         shift_pre_hatch=".",
    #         hold_hatch="/",
    #         hold_pre_hatch="/",
    #         bc_hatch="x",
    #         alpha_bc=0.2,
    #         linewidth=2,
    #     ),
    #     plot=False,
    # )
    #
    # plt.show()
