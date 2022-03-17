import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from tqdm import tqdm

from conv_ssl.plot_utils import plot_vad_oh, plot_pr_curve
from conv_ssl.utils import everything_deterministic, to_device
from conv_ssl.evaluation.evaluation import test
from conv_ssl.evaluation.k_fold_aggregate import (
    discrete,
    independent,
    test_single_model,
)
from conv_ssl.evaluation.utils import load_dm, load_model

event_kwargs = dict(
    shift_onset_cond=1,
    shift_offset_cond=1,
    hold_onset_cond=1,
    hold_offset_cond=1,
    min_silence=0.15,
    non_shift_horizon=2.0,
    non_shift_majority_ratio=0.95,
    metric_pad=0.05,
    metric_dur=0.1,
    metric_onset_dur=0.2,
    metric_pre_label_dur=0.5,
    metric_min_context=1.0,
    bc_max_duration=1.0,
    bc_pre_silence=1.0,
    bc_post_silence=2.0,
)


def plot_batch(
    probs, events, vad, metric, n_rows=4, n_cols=4, plw=2, alpha_vad=0.6, plot=True
):
    valid = ["hold", "shift", "predict_shift", "predict_bc", "short", "long"]
    vad = vad.cpu()

    assert metric in valid, f"{metric} is not valid: {valid}"

    fig, ax = plt.subplots(n_rows, n_cols, sharey=True, sharex=True, figsize=(16, 8))
    b = 0
    for row in range(n_rows):
        for col in range(n_cols):
            _ = plot_vad_oh(vad[b], ax=ax[row, col], alpha=alpha_vad)
            if metric in ["hold", "shift"]:
                _ = plot_vad_oh(
                    events["shift"][b].cpu(),
                    ax=ax[row, col],
                    colors=["g", "g"],
                    alpha=0.5,
                )
                _ = plot_vad_oh(
                    events["hold"][b].cpu(),
                    ax=ax[row, col],
                    colors=["r", "r"],
                    alpha=0.5,
                )
                ax[row, col].plot(probs["p"][b, :, 0], linewidth=plw, color="darkblue")
                ax[row, col].plot(
                    probs["p"][b, :, 1], linewidth=plw, color="darkorange"
                )
            elif metric == "predict_bc":
                _ = plot_vad_oh(
                    events["predict_bc_pos"][b].cpu(),
                    ax=ax[row, col],
                    colors=["g", "g"],
                    alpha=0.5,
                )
                _ = plot_vad_oh(
                    events["predict_bc_neg"][b].cpu(),
                    ax=ax[row, col],
                    colors=["r", "r"],
                    alpha=0.5,
                )
                ax[row, col].plot(
                    probs["bc_prediction"][b, :, 0].cpu(),
                    linewidth=plw,
                    color="darkblue",
                )
                ax[row, col].plot(
                    probs["bc_prediction"][b, :, 1].cpu(),
                    linewidth=plw,
                    color="darkorange",
                )
            elif metric == "predict_shift":
                _ = plot_vad_oh(
                    events["predict_shift_pos"][b].cpu(),
                    ax=ax[row, col],
                    colors=["g", "g"],
                    alpha=0.5,
                )
                _ = plot_vad_oh(
                    events["predict_shift_neg"][b].cpu(),
                    ax=ax[row, col],
                    colors=["r", "r"],
                    alpha=0.5,
                )

                pre_probs = probs.get("pre_probs", None)
                if pre_probs is not None:
                    ax[row, col].plot(
                        probs["pre_probs"][b, :, 0].cpu(),
                        linewidth=plw,
                        color="darkblue",
                    )
                    ax[row, col].plot(
                        probs["pre_probs"][b, :, 1].cpu(),
                        linewidth=plw,
                        color="darkorange",
                    )
                else:
                    ax[row, col].plot(
                        probs["p"][b, :, 0].cpu(), linewidth=plw, color="darkblue"
                    )
                    ax[row, col].plot(
                        probs["p"][b, :, 1].cpu(), linewidth=plw, color="darkorange"
                    )
            elif metric in ["short", "long"]:
                _ = plot_vad_oh(
                    events["short"][b].cpu(),
                    ax=ax[row, col],
                    colors=["g", "g"],
                    alpha=0.5,
                )
                _ = plot_vad_oh(
                    events["long"][b].cpu(),
                    ax=ax[row, col],
                    colors=["r", "r"],
                    alpha=0.5,
                )
                pre_probs = probs.get("pre_probs", None)
                if pre_probs is not None:
                    ax[row, col].plot(
                        probs["pre_probs"][b, :, 0].cpu(),
                        linewidth=plw,
                        color="darkblue",
                    )
                    ax[row, col].plot(
                        probs["pre_probs"][b, :, 1].cpu(),
                        linewidth=plw,
                        color="darkorange",
                    )
                else:
                    # ax[row, col].plot(
                    #     probs["p"][b, :, 0].cpu(), linewidth=plw, color="darkblue"
                    # )
                    # ax[row, col].plot(
                    #     probs["p"][b, :, 1].cpu(), linewidth=plw, color="darkorange"
                    # )
                    ax[row, col].plot(
                        probs["bc_prediction"][b, :, 0].cpu(),
                        linewidth=plw,
                        color="darkblue",
                    )
                    ax[row, col].plot(
                        probs["bc_prediction"][b, :, 1].cpu(),
                        linewidth=plw,
                        color="darkorange",
                    )

            b += 1
            if b == vad.shape[0]:
                break
        if b == vad.shape[0]:
            break
    plt.tight_layout()
    if plot:
        plt.pause(0.1)

    return fig, ax


def test_single_model(
    run_path,
    event_kwargs,
    threshold_pred_shift=0.5,
    threshold_short_long=0.5,
    threshold_bc_pred=0.1,
    bc_pred_pr_curve=False,
    shift_pred_pr_curve=False,
    long_short_pr_curve=False,
    batch_size=16,
):
    "g" "test model" ""

    # Load data (same across folds)
    dm = load_dm(vad_hz=100, horizon=2, batch_size=batch_size, num_workers=4)

    # Load model and process test-set
    model = load_model(run_path=run_path, eval=True, strict=False)

    if torch.cuda.is_available():
        model = model.to("cuda")

    # Updatemetric_kwargs metrics
    # for metric, val in metric_kwargs.items():
    #     model.conf["vad_projection"][metric] = val

    model.test_metric = model.init_metric(
        model.conf,
        model.frame_hz,
        threshold_pred_shift=threshold_pred_shift,
        threshold_short_long=threshold_short_long,
        threshold_bc_pred=threshold_bc_pred,
        bc_pred_pr_curve=bc_pred_pr_curve,
        shift_pred_pr_curve=shift_pred_pr_curve,
        long_short_pr_curve=long_short_pr_curve,
        **event_kwargs,
    )
    model.test_metric = model.test_metric.to(model.device)

    result = test(model, dm, online=False)
    return result, model


def PRCURVE():
    # run_path = independent["10"]
    run_path = discrete["3"]
    # Aggregate
    res, model = test_single_model(
        run_path,
        event_kwargs=event_kwargs,
        threshold_pred_shift=0.5,
        threshold_short_long=0.5,
        threshold_bc_pred=0.1,
        bc_pred_pr_curve=True,
        shift_pred_pr_curve=True,
        long_short_pr_curve=True,
        batch_size=16,
    )
    res = res[0]
    result = model.test_metric.compute()
    model = model.to("cuda")
    for k, v in res.items():
        print(k, v)

    # torch.save(result, f"ind_{run_path}.pt")
    torch.save(result, f"dis_{run_path}.pt")

    fig, ax = plot_pr_curve(
        precision=result["pr_curve_bc_pred"][0],
        recall=result["pr_curve_bc_pred"][1],
        thresholds=result["pr_curve_bc_pred"][2],
        model="ind-BC-Pred",
        color_auc_max="b",
        thresh_min=0.01,
        plot=False,
    )
    fig1, ax1 = plot_pr_curve(
        precision=result["pr_curve_shift_pred"][0],
        recall=result["pr_curve_shift_pred"][1],
        thresholds=result["pr_curve_shift_pred"][2],
        model="ind-Shift-Pred",
        color_auc_max="b",
        thresh_min=0.01,
        plot=False,
    )
    fig2, ax2 = plot_pr_curve(
        precision=result["pr_curve_long_short"][0],
        recall=result["pr_curve_long_short"][1],
        thresholds=result["pr_curve_long_short"][2],
        model="ind-Long/Short",
        color_auc_max="b",
        thresh_min=0.01,
        plot=False,
    )
    plt.pause(0.1)


def batch_view():
    # Model
    run_path = discrete["3"]
    model = load_model(run_path=run_path, eval=True, strict=False)
    model = model.eval()
    model.test_metric = model.init_metric(
        model.conf, model.frame_hz, bc_pred_pr_curve=False, **event_kwargs
    )
    model.test_metric = model.test_metric.to(model.device)

    dm = load_dm(batch_size=16)
    diter = iter(dm.test_dataloader())

    batch = next(diter)
    batch = to_device(batch)

    # BATCH
    with torch.no_grad():
        events = model.test_metric.extract_events(batch["vad"])
        _, out, batch = model.shared_step(batch)
        probs = model.get_next_speaker_probs(out["logits_vp"], batch["vad"])
        for k, v in events.items():
            events[k] = v[:, :1000]
        # for k, v in events.items():
        #     print(k, v.shape)
    fig, ax = plot_batch(probs, events, batch["vad"], metric="predict_bc")
    fig, ax = plot_batch(probs, events, batch["vad"], metric="predict_shift")
    fig, ax = plot_batch(probs, events, batch["vad"], metric="long")


def extract_loss_curve():

    from conv_ssl.evaluation.evaluation import SymmetricSpeakersCallback

    run_path = discrete["3"]
    # Load data (same across folds)
    dm = load_dm(vad_hz=100, horizon=2, batch_size=16, num_workers=4)
    # Load model and process test-set
    model = load_model(run_path=run_path, eval=True, strict=False)
    # if torch.cuda.is_available():
    #     model = model.to("cuda")

    # model.test_metric = model.init_metric(
    #     model.conf,
    #     model.frame_hz,
    #     # threshold_pred_shift=threshold_pred_shift,
    #     # threshold_short_long=threshold_short_long,
    #     # threshold_bc_pred=threshold_bc_pred,
    #     # bc_pred_pr_curve=bc_pred_pr_curve,
    #     # shift_pred_pr_curve=shift_pred_pr_curve,
    #     # long_short_pr_curve=long_short_pr_curve,
    #     **event_kwargs,
    # )
    # model.test_metric = model.test_metric.to(model.device)

    trainer = pl.Trainer(
        gpus=-1,
        deterministic=True,
        logger=None,
        callbacks=[SymmetricSpeakersCallback()],
    )
    result = trainer.test(model, dataloaders=dm.val_dataloader(), verbose=False)

    loss_vector = model.loss_vector / model.loss_n

    torch.save(loss_vector, "loss_vector.pt")

    n = 0
    losses = torch.zeros(1000, dtype=torch.float)
    dloader = dm.val_dataloader()
    for batch in tqdm(dloader, total=len(dloader)):
        loss, _, _ = model.shared_step(to_device(batch, model.device), reduction="none")
        n += loss["vp"].shape[0]
        losses += loss["vp"].sum(0).cpu()
        print(losses.device)
    losses /= n

    fig, ax = plt.subplots(1, 1)
    ax.plot()
    result = test(model, dm, online=False)


if __name__ == "__main__":

    pl.seed_everything(100)
    everything_deterministic()

    extract_loss_curve()
