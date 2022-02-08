import matplotlib.pyplot as plt
import torch
from conv_ssl.model import VPModel
from conv_ssl.evaluation import load_dm, to_device


torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True)


if __name__ == "__main__":

    chpt = "checkpoints/w2v/epoch=80.ckpt"
    model = VPModel.load_from_checkpoint(chpt)
    # model.to("cuda")
    model.eval()
    optim = model.configure_optimizers()
    step = 500
    optim.zero_grad()
    batch = to_device(next(diter), model.device)
    batch["waveform"].requires_grad = True
    loss, out, batch = model.shared_step(batch, reduction="none")
    loss = loss["vp"]
    # backward
    loss[:, step].norm().backward()
    g = batch["waveform"].grad.abs()
    b = 0
    fig, ax = plt.subplots(1, 1)
    fig.suptitle("Wav2vec")
    ax.plot(g[b].cpu() / g[b].max())
    plt.pause(0.1)

    ###############################################
    # Wav2vec
    ###############################################
    conf = VPModel.load_config()
    conf["encoder"]["output_layer"] = 1
    model = VPModel(conf)
    optim = model.configure_optimizers()
    ##############################################
    dm = load_dm(model)
    diter = iter(dm.val_dataloader())
    ##############################################
    step = 500
    optim.zero_grad()
    batch = to_device(next(diter), model.device)
    batch["waveform"].requires_grad = True
    loss, out, batch = model.shared_step(batch, reduction="none")
    loss = loss["vp"]
    # backward
    loss[:, step].norm().backward()
    g = batch["waveform"].grad.abs()
    b = 0
    fig, ax = plt.subplots(1, 1)
    fig.suptitle("Wav2vec")
    ax.plot(g[b].cpu() / g[b].max())
    plt.pause(0.1)

    ###############################################
    # VQ-Wav2vec
    ###############################################
    conf = VPModel.load_config()
    conf["encoder"]["type"] = "vq_wav2vec"
    conf["encoder"]["output_layer"] = 1
    model = VPModel(conf)
    optim = model.configure_optimizers()
    ##############################################
    dm = load_dm(model)
    diter = iter(dm.val_dataloader())
    ##############################################
    step = 500
    optim.zero_grad()
    batch = to_device(next(diter), model.device)
    batch["waveform"].requires_grad = True
    loss, out, batch = model.shared_step(batch, reduction="none")
    loss = loss["vp"]
    # backward
    loss[:, step].norm().backward()
    g = batch["waveform"].grad.abs()
    b = 0
    fig, ax = plt.subplots(1, 1)
    fig.suptitle("VQ-Wav2Vec")
    ax.plot(g[b].cpu() / g[b].max())
    plt.pause(0.1)

    ###############################################
    # CPC
    ###############################################
    conf = VPModel.load_config()
    conf["encoder"]["type"] = "cpc"
    conf["ar"]["dim"] = 256
    model = VPModel(conf)
    optim = model.configure_optimizers()
    ##############################################
    dm = load_dm(model)
    diter = iter(dm.val_dataloader())
    ##############################################
    step = 500
    optim.zero_grad()
    batch = to_device(next(diter), model.device)
    batch["waveform"].requires_grad = True
    loss, out, batch = model.shared_step(batch, reduction="none")
    loss = loss["vp"]
    # backward
    loss[:, step].norm().backward()
    g = batch["waveform"].grad.abs()
    b = 0
    fig, ax = plt.subplots(1, 1)
    fig.suptitle("CPC")
    ax.plot(g[b].cpu() / g[b].max())
    plt.pause(0.1)
