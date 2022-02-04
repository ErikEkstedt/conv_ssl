import math
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer, Callback

from conv_ssl.model import VPModel, ShiftHoldMetric
from conv_ssl.plot_utils import plot_next_speaker_probs, plot_all_labels
from datasets_turntaking import DialogAudioDM
from datasets_turntaking.features.vad import VAD, VadProjection, DialogEvents


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
    vad_projection_head = VadProjection()
    print("on_silent_shift: ", tuple(vad_projection_head.on_silent_shift.shape))
    print("on_silent_hold: ", tuple(vad_projection_head.on_silent_hold.shape))
    print("on_active_shift: ", tuple(vad_projection_head.on_active_shift.shape))
    print("on_active_hold: ", tuple(vad_projection_head.on_active_hold.shape))
    print("--------------------------------------------")
    plot_all_labels(vad_projection_head, next_speaker=0)


def load_dm(model, test=False, vad_history=None):
    data_conf = DialogAudioDM.load_config()

    if vad_history is not None:
        data_conf["dataset"]["vad_history"] = vad_history

    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hz=model.frame_hz,
        vad_bin_times=model.conf["vad_class_prediction"]["bin_times"],
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=4,
        num_workers=4,
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


if __name__ == "__main__":

    # from conv_ssl.eval import manual_eval
    import matplotlib as mpl

    mpl.use("TkAgg")

    # TODO: load artifact from wandb

    # chpt = "checkpoints/wav2vec_epoch=6-val_loss=2.42136.ckpt"
    # chpt = "checkpoints/hubert_epoch=18-val_loss=1.61074.ckpt"
    # chpt = "checkpoints/w2v/epoch=29-step=28229.ckpt"
    chpt = "checkpoints/w2v/epoch=80-step=76220.ckpt"
    model = VPModel.load_from_checkpoint(chpt)
    model.to("cuda")
    model.eval()
    dm = load_dm(model, test=False)

    # Single Plots
    # diter = iter(dm.val_dataloader())
    # batch = next(diter)
    # plot_batch(model, batch)

    # evaluation
    dloader = dm.val_dataloader()

    callbacks = []
    callbacks = [AblationCallback(vad_history="reverse")]

    trainer = Trainer(gpus=-1, limit_test_batches=200, callbacks=callbacks)
    # trainer = Trainer(gpus=-1, limit_test_batches=200)
    result = trainer.test(model, dataloaders=dloader, verbose=False)
    result = result[0]
    result

    # dloader = dm.test_dataloader()
    # single_metric
    result = evaluation(model, dloader, event_start_pad=5, event_target_duration=1)
    # val: pad=5, dur=1
    #  'test_loss': 2.0851874351501465,
    #  'test_loss_vp': 2.0851874351501465,
    #  'test/f1_weighted': 0.9111487865447998,
    #  'test/f1_shift': 0.7565796375274658,
    #  'test/shift_precision': 0.7849553823471069,
    #  'test/shift_recall': 0.7301838397979736,
    #  'test/shift_support': 5059.0,
    #  'test/f1_hold': 0.9465782642364502,
    #  'test/hold_precision': 0.9391277432441711,
    #  'test/hold_recall': 0.95414799451828,
    #  'test/hold_support': 22071.0
    tot = result["test/shift_support"] + result["test/hold_support"]

    # VAD - reverse
    # 'test_loss': 9.88
    # 'test/f1_weighted': 0.8895

    # VAD - zero: (no events)
    # 'test_loss': 3.957580804824829,

    # VADHISTORY - reverse
    # 'test_loss': 2.1889231204986572,
    # 'test/f1_weighted': 0.8931927680969238,

    # vad-reverse, vad-history-equal
    # 'test_loss': 10.395157814025879,
    # 'test/f1_weighted': 0.9208936095237732,
    #  'test/f1_shift': 0.7450224757194519,
    #  'test/shift_precision': 0.7277289628982544,
    #  'test/shift_recall': 0.7631579041481018,
    #  'test/shift_support': 3040.0,
    #  'test/f1_hold': 0.9526422619819641,
    #  'test/hold_precision': 0.9568655490875244,
    #  'test/hold_recall': 0.9484560489654541,
    #  'test/hold_support': 16840.0}

    # VAD History
    # Normal
    #  'test/f1_weighted': 0.9286942481994629,
    # Equal vad_history
    #  'test/f1_weighted': 0.9206141233444214,
    # reverse vad_history
    #  'test/f1_weighted': 0.890008807182312,

    # {'test_loss': 2.076958179473877,
    #  'test_loss_vp': 2.076958179473877,
    #  'test/f1_weighted': 0.9271664023399353,
    #  'test/f1_shift': 0.7585858702659607,
    #  'test/shift_precision': 0.7768965363502502,
    #  'test/shift_recall': 0.7411184310913086,
    #  'test/shift_support': 3040.0,
    #  'test/f1_hold': 0.9575990438461304,
    #  'test/hold_precision': 0.9536513686180115,
    #  'test/hold_recall': 0.9615795612335205,
    #  'test/hold_support': 16840.0}
    ##################################################
    # NO VAD-HISTORY
    ##################################################
    # {'test_loss': 4.358254432678223,
    #  'test_loss_vp': 4.358254432678223,
    #  'test/f1_weighted': 0.5671782493591309,
    #  'test/f1_shift': 0.32110029458999634,
    #  'test/shift_precision': 0.20325487852096558,
    #  'test/shift_recall': 0.7641447186470032,
    #  'test/shift_support': 3040.0,
    #  'test/f1_hold': 0.611600935459137,
    #  'test/hold_precision': 0.9151579737663269,
    #  'test/hold_recall': 0.4592636525630951,
    #  'test/hold_support': 16840.0}

    # TODO: extract all eval from single forward (no need to forward again and again, that's silly)
    # aggregate metrics
    agg_res = metric_aggregation(
        model, dloader, pads=[10, 20, 60], durs=[1, 10, 20, 60]
    )

    pad_dur = []
    total_events = []
    f1w = []
    shift_ratio = []
    for pad_name, metric_durs in agg_res.items():
        # p = int(pad_name.split("_")[-1])
        tmp_f1 = []
        tmp_rat = []
        total = []
        for dur_name, m in metric_durs.items():
            # d = int(dur_name.split("_")[-1])
            # F1
            f1w_tmp = m["test/f1_weighted"]
            tmp_f1.append(f1w_tmp)
            # shift-ratio
            tot = m["test/shift_support"] + m["test/hold_support"]
            rs = m["test/shift_support"] / tot
            tmp_rat.append(rs)
            total.append(tot)
        f1w.append(tmp_f1)
        shift_ratio.append(tmp_rat)
    f1w = torch.tensor(f1w)
    shift_ratio = torch.tensor(shift_ratio)
    total = torch.tensor(total)
    print(f1w)
    print(shift_ratio)
    print(total)

    # Plot result: matplots, table
    # Save result: checkpoint, model.conf
    # torch.save(agg_res, result_path)

    agg_res = torch.load("aggregate_result_1000.pt")

    # ratio
    rs = result["test/shift_support"] / (
        result["test/shift_support"] + result["test/hold_support"]
    )
    rh = 1 - rs

    print("F1 weighted: ", result["test/f1_weighted"])
    print("F1 shift: ", result["test/f1_shift"])
    print(f"Shifts: {round(rs*100, 2)}%")
    print(f"Holds: {round(rh*100, 2)}%")

    # f1 = 2 * 0.949 * 0.963 / (0.949 + 0.963)

    # f1
    # [0.9180, 0.9118, 0.9151, 0.9050],
    # [0.9122, 0.9150, 0.9160, 0.9078],
    # [0.9035, 0.9008, 0.9091, 0.8789]

    # shift ratio
    # [0.1633, 0.1745, 0.1620, 0.1586],
    # [0.1727, 0.1620, 0.1543, 0.1617],
    # [0.1541, 0.1586, 0.1617, 0.1706]
    # events
    # [52197.,  8847.,  7007.,  2157.],
    # [ 8639.,  7007.,  5470.,  1540.],
    # [2888.0, 2157.0, 1540.0, 469.0]
