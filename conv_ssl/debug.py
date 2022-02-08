from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib as mpl

from tqdm import tqdm
import torch
import pytorch_lightning as pl

from conv_ssl.ulm_projection import ULMProjection
from conv_ssl.models import ProjectionMetricCallback
from datasets_turntaking.dm_dialog_audio import DialogAudioDM, print_dm
from conv_ssl.utils import count_parameters


mpl.use("tkagg")

# TODO: Plot stats
# TODO: Plot stats REGRESSION
# TODO: Simplify callback, prepare metrics for ActivityProjectionHead


def to_device(batch, device="cuda"):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def init(regression=False):
    parser = ArgumentParser()
    parser = ULMProjection.add_model_specific_args(parser)
    parser = DialogAudioDM.add_data_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name_info", type=str, default="")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--log_gradients", action="store_true")
    parser.add_argument("--animation_epoch_start", default=10, type=int)
    parser.add_argument("--animation_n", default=10, type=int)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    data_conf = DialogAudioDM.load_config(path=args.data_conf, args=args)
    print_dm(data_conf, args)
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_include_ratio=data_conf["dataset"]["audio_include_ratio"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hop_time=data_conf["dataset"]["vad_hop_time"],
        vad_bin_sizes=data_conf["dataset"]["vad_bin_sizes"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.prepare_data()
    dm.setup()

    conf = ULMProjection.load_config(path=args.conf, args=args)
    conf["vad_class_prediction"]["regression"] = regression
    model = ULMProjection(conf)
    # Name the run e.g. hubert_44_41
    name = conf["encoder"]["type"].replace("_base", "")
    name += f"_{conf['tier1']['num_layers']}{conf['tier1']['num_heads']}"
    name += f"_{conf['tier2']['num_layers']}{conf['tier2']['num_heads']}"
    print("-" * 60)
    print(f"Model Name: {name}")
    print("Base: ", args.conf)
    print("PARAMETERS: ", count_parameters(model))
    print()

    model = model.to("cuda")
    optim = model.configure_optimizers()
    return model, optim, dm


def fit(model, dm, epochs, N, regression=False, val=True):
    # Callbacks & Logger
    callback = ProjectionMetricCallback()
    K = 5
    if regression:
        K = 1
    stats = {
        "f1": [],
        "f1_weighted": [],
        "hold": [[] for _ in range(K)],
        "shift": [[] for _ in range(K)],
        "class_silence": [[] for _ in range(K)],
        "hold_n": [],
        "shift_n": [],
    }
    for epoch in tqdm(range(epochs)):
        for bidx, batch in enumerate(dm.val_dataloader()):
            batch = to_device(batch)
            loss = model.training_step(batch, batch_idx=bidx)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if bidx == N:
                break

        if val:
            with torch.no_grad():
                callback.on_validation_epoch_start()
                for bidx, batch in enumerate(dm.val_dataloader()):
                    batch = to_device(batch)
                    out = model.validation_step(batch, batch_idx=bidx)
                    out = to_device(out, "cpu")
                    callback.on_validation_batch_end(
                        trainer=None,
                        pl_module=model,
                        outputs=out,
                        batch=batch,
                        batch_idx=bidx,
                        dataloader_idx=0,
                    )
                    if bidx == N:
                        break
                callback.on_validation_epoch_end(trainer=None, pl_module=model)
                for k, v in callback.val_metric.compute().items():
                    if k in stats:
                        if k in [
                            "hold",
                            "shift",
                            "next_speaker_silence",
                            "class_silence",
                        ]:
                            for i in range(K):
                                stats[k][i].append(v[i])
                        else:
                            stats[k].append(v)
    if val:
        for k, v in stats.items():
            if isinstance(stats[k][0], torch.Tensor):
                stats[k] = torch.stack(v)
            elif isinstance(stats[k][0], list):
                for i in range(K):
                    stats[k][i] = torch.stack(v[i])
                stats[k] = torch.stack(stats[k])
            else:
                stats[k] = torch.tensor(v)
    return stats, model, callback


if __name__ == "__main__":
    for regression in [True, False]:
        model, optim, dm = init(regression=regression)
        stats, model, callback = fit(
            model, dm, epochs=20, N=20, regression=regression, val=True
        )
        n = len(stats) - 2
        fig, ax = plt.subplots(n, 1, sharex=True, figsize=(16, 9))
        i = 0
        for k, v in stats.items():
            if k in ["hold_n", "shift_n"]:
                continue
            else:
                if v.ndim == 2:
                    for vv in v:
                        ax[i].plot(vv)
                else:
                    ax[i].plot(v)
                ax[i].set_ylabel(k)
                ax[i].set_ylim([0, 1])
                i += 1
        plt.tight_layout()
        plt.pause(0.01)
    input()
