from os import makedirs, environ

import torch
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import WandbLogger

from conv_ssl.evaluation.utils import load_model, load_dm
from conv_ssl.utils import everything_deterministic

# from vad_turn_taking import DialogEvents, ProjectionCodebook
# import matplotlib.pyplot as plt
# from conv_ssl.plot_utils import plot_next_speaker_probs, plot_all_labels, plot_window
# import matplotlib as mpl
# mpl.use("TkAgg")


everything_deterministic()

# TODO: make the test such that it uploads to wandb (not lightning logs) to get all results in one place


def get_symmetric_batch(batch):
    """Appends a flipped version of the batch-samples"""
    for k, v in batch.items():
        if k == "vad":
            flipped = torch.stack((v[..., 1], v[..., 0]), dim=-1)
        elif k == "vad_history":
            flipped = 1.0 - v
        else:
            flipped = v
        if isinstance(v, torch.Tensor):
            batch[k] = torch.cat((v, flipped))
        else:
            batch[k] = v + flipped
    return batch


class SymmetricSpeakersCallback(Callback):
    """
    This callback "flips" the speakers such that we get a fair evaluation not dependent on the
    biased speaker-order / speaker-activity

    The audio is mono which requires no change.

    The only change we apply is to flip the channels in the VAD-tensor and get the corresponding VAD-history
    which is defined as the ratio of speaker 0 (i.e. vad_history_flipped = 1 - vad_history)
    """

    def on_test_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = get_symmetric_batch(batch)

    def on_val_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        batch = get_symmetric_batch(batch)


def test(model, dm, max_batches=None, project="VPModelTest", online=True):
    logger = None
    if online:
        savedir = "runs/" + project
        makedirs(savedir, exist_ok=True)
        logger = WandbLogger(
            save_dir=savedir,
            project=project,
            name=model.run_name,
            log_model=False,
        )

    if max_batches is not None:
        trainer = Trainer(
            gpus=-1,
            limit_test_batches=max_batches,
            deterministic=True,
            logger=logger,
            callbacks=[SymmetricSpeakersCallback()],
        )
    else:
        trainer = Trainer(
            gpus=-1,
            deterministic=True,
            logger=logger,
            callbacks=[SymmetricSpeakersCallback()],
        )
    result = trainer.test(model, dataloaders=dm.test_dataloader(), verbose=False)
    return result


if __name__ == "__main__":

    # run_path: found in wandb information tab
    run_path = "how_so/VPModel/djtjou83"  # discrete
    # run_path = "how_so/VPModel/27ly86w3"  # independent (same bin size)
    # run_path = "how_so/VPModel/27ly86w3"  # Comparative
    model = load_model(run_path=run_path)
    dm = load_dm(model, batch_size=4, num_workers=4)

    result = test(model, dm, online=False)
    # batch = next(iter(dm.val_dataloader()))

    from conv_ssl.plot_utils import plot_window
    from vad_turn_taking.vad import DialogEvents, VAD
    import matplotlib.pyplot as plt

    frame_start_pad = 5
    frame_target_duration = 10
    frame_horizon = 200
    frame_min_context = 100
    frame_min_duration = frame_start_pad + frame_target_duration
    frame_pre_active = 50
    bc_pre_silence_frames = 150
    bc_post_silence_frames = 300
    bc_max_active_frames = 100
    ###########################################

    diter = iter(dm.val_dataloader())

    with torch.no_grad():
        batch = next(diter)
        # batch = get_symmetric_batch(batch)
        loss, out, batch = model.shared_step(batch)
        probs = out["logits_vp"].softmax(-1)
    #####################################################
    # BACKCHANNEL Prob
    #####################################################
    bc_idx = model.projection_codebook.on_active_bc
    # bc_idx = model.projection_codebook.on_active_overlap
    bc_oh = model.projection_codebook.idx_to_onehot(bc_idx)
    ap = probs[..., bc_idx[0]].sum(-1)
    bp = probs[..., bc_idx[1]].sum(-1)
    p = torch.stack((ap, bp), dim=-1)
    ds = VAD.vad_to_dialog_vad_states(batch["vad"])
    bc_a = torch.where(ds == 3)  # speaker B active
    bc_a = [bc_a[0], bc_a[1], torch.zeros(bc_a[0].shape, dtype=torch.long)]
    bc_b = torch.where(ds == 0)  # speaker A active
    bc_b = [bc_b[0], bc_b[1], torch.ones(bc_b[0].shape, dtype=torch.long)]
    bc_probs = torch.zeros_like(batch["vad"])
    bc_probs[bc_a] = p[bc_a[0], bc_a[1], 0]
    bc_probs[bc_b] = p[bc_b[0], bc_b[1], 1]
    # idx = model.projection_codebook.on_silent_competition
    # oh = model.projection_codebook.idx_to_onehot(idx)
    # p = probs[..., idx].sum(-1) * 20
    # ds = VAD.vad_to_dialog_vad_states(batch["vad"])
    # w = torch.where(ds == 1)  # mutual silence
    # bc_probs = torch.zeros_like(batch["vad"])
    # bc_probs[w] = torch.stack((p[w], p[w]), dim=-1)
    #########################################################
    for b in range(batch["vad"].shape[0]):
        z = torch.zeros_like(batch["vad"][b])
        fig, ax = plot_window(
            probs=bc_probs[b],
            vad=batch["vad"][b],
            hold=z,
            shift=z,
            pre_hold=z,
            pre_shift=z,
            backchannels=z,
            only_over_05=False,
        )
    plt.show()

    hold, shift = DialogEvents.on_silence(
        batch["vad"],
        start_pad=frame_start_pad,
        target_frames=frame_target_duration,
        horizon=frame_horizon,
        min_context=frame_min_context,
        min_duration=frame_min_duration,
    )
    pre_hold, pre_shift = DialogEvents.get_active_pre_events(
        batch["vad"],
        hold,
        shift,
        start_pad=frame_start_pad,
        active_frames=frame_pre_active,
        min_context=frame_min_context,
    )
    backchannels = DialogEvents.extract_bc_candidates(
        batch["vad"],
        pre_silence_frames=bc_pre_silence_frames,
        post_silence_frames=bc_post_silence_frames,
        max_active_frames=bc_max_active_frames,
    )
