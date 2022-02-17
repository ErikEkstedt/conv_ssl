import math
import matplotlib.pyplot as plt
from os.path import basename, dirname, exists, join

import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer, Callback

from conv_ssl.utils import to_device
from conv_ssl.plot_utils import plot_next_speaker_probs, plot_all_labels, plot_window
from vad_turn_taking import DialogEvents, ProjectionCodebook


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


def old():

    # TODO: load artifact from wandb

    checkpoints = [
        # "checkpoints/cpc/cpc_04_14.ckpt",
        # "checkpoints/cpc/cpc_04_24.ckpt",
        # "checkpoints/cpc/cpc_04_44.ckpt",
        "checkpoints/cpc/cpc_04_14_reg.ckpt",
        "checkpoints/cpc/cpc_04_24_reg.ckpt",
        "checkpoints/cpc/cpc_04_44_reg.ckpt",
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
    # torch.save(results, "assets/evaluation_scores.pt")
    # r = torch.load("assets/evaluation_scores.pt")

    r = torch.load("evaluation_reg_scores.pt")

    # #######################################################
    chpt = "checkpoints/cpc/cpc_04_44_reg.ckpt"
    chpt = "checkpoints/cpc/cpc_04_44.ckpt"
    model = VPModel.load_from_checkpoint(chpt)
    model = model.to("cuda")
    model.eval()
    dm = load_dm(model, test=False)
    diter = iter(dm.val_dataloader())

    # trainer = Trainer(gpus=-1, limit_test_batches=50, deterministic=True)
    # result = trainer.test(model, dataloaders=dm.val_dataloader(), verbose=False)
    # print(result[0])

    print(model.val_metric)
    batch = next(diter)
    batch = to_device(batch, model.device)
    model.val_metric.reset()
    with torch.no_grad():
        loss, out, batch = model.shared_step(batch)
        # probs = out['logits_vp'].softmax(dim=-1).view(-1, 256)
        # p_log_p = probs*probs.log()
        # ent = -p_log_p.sum(-1)

        next_probs, pre_probs = model.get_next_speaker_probs(
            out["logits_vp"], vad=batch["vad"]
        )
        events = model.val_metric.extract_events(batch["vad"])
        model.val_metric.update(
            next_probs, vad=batch["vad"], events=events, bc_pre_probs=pre_probs
        )
        r = model.val_metric.compute()
        if model.regression:
            p = out["logits_vp"].sigmoid().cpu()
        else:
            p = out["logits_vp"].topk(dim=-1, k=3).indices
            p = p.permute(2, 0, 1)
            p = model.projection_codebook.idx_to_onehot(p).cpu()

    def plot_regression_prediction(
        p,
        step,
        ax,
        bin_times=[0.2, 0.4, 0.6, 0.8],
        lw=2,
        ls=None,
        current=True,
        fill=False,
    ):
        """
        regression prediction lines
        """
        # add zero at start to include values from prediction step
        x = torch.tensor([0] + bin_times).cumsum(dim=0)
        x += step
        # tmax = x[-1]

        # replicate first bin to match x
        ap = p[step, 0]
        ap = torch.cat((ap[:1], ap))

        # replicate first bin to match x
        bp = p[step, 1] - 1
        bp = torch.cat((bp[:1], bp))

        ret = {}
        if current:
            ret["current"] = ax.vlines(step, ymin=-1, ymax=1, color="r", linewidth=3)
        ret["a_pred"] = ax.step(
            x, ap, where="pre", linewidth=lw, linestyle=ls, color="b"
        )[0]
        ret["b_pred"] = ax.step(
            x, bp, where="pre", linewidth=lw, linestyle=ls, color="orange"
        )[0]
        return ret

    b = 0
    fig, ax = plot_window(
        next_probs[b].cpu(),
        vad=batch["vad"][b].cpu(),
        hold=events["hold"][b].cpu(),
        shift=events["shift"][b].cpu(),
        pre_hold=events["pre_hold"][b].cpu(),
        pre_shift=events["pre_shift"][b].cpu(),
        backchannels=events["backchannels"][b].cpu(),
        plot_kwargs=dict(
            alpha_event=0.1,
            alpha_vad=0.1,
            alpha_probs=0.1,
            shift_hatch=".",
            shift_pre_hatch=".",
            hold_hatch="/",
            hold_pre_hatch="/",
            bc_hatch="x",
            alpha_bc=0.1,
            linewidth=2,
        ),
        plot=False,
    )
    if model.regression:
        for step in range(0, 1000):
            lines = plot_regression_prediction(
                p[b], step=step, ax=ax[0], bin_times=[20, 40, 60, 80], lw=2
            )
            plt.pause(0.01)
            lines["current"].remove()
            lines["a_pred"].remove()
            lines["b_pred"].remove()
    else:
        for step in range(0, 1000):
            all_lines = []
            for k, ls in zip(range(3), ["solid", "dashed", "dotted"]):
                lines = plot_regression_prediction(
                    p[k, b],
                    step=step,
                    ax=ax[0],
                    bin_times=[20, 40, 60, 80],
                    lw=2,
                    ls=ls,
                    current=k == 0,
                )
                all_lines.append(lines)
            plt.pause(0.01)
            for lines in all_lines:
                if "current" in lines:
                    lines["current"].remove()
                lines["a_pred"].remove()
                lines["b_pred"].remove()

    plt.close("all")

    lines["current"]

    plt.pause(0.1)

    step = 400
    fig2, ax2 = plt.subplots(1, 1)
    ax2 = plot_regression_prediction(
        p[b], step=step, ax=ax2, bin_times=[20, 40, 60, 80]
    )
    plt.pause(0.1)

    plt.show()