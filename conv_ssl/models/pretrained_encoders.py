from os.path import join, exists, dirname
from os import makedirs

# from numpy import printoptions
import torch
from torchaudio.pipelines import HUBERT_BASE, WAV2VEC2_BASE
from torchaudio.models import hubert_base, wav2vec2_base


from conv_ssl.utils import repo_root
from conv_ssl.models.WavLM import WavLM, WavLMConfig


"""
pipeline downloads to `~/.cache/torch/hub/checkpoints/`
"""


CHECKPOINTS = {
    "hubert_base": join(
        repo_root(), "assets/checkpoints/hubert/hubert_fairseq_base_ls960.pth"
    ),
    "wav2vec2_base": join(
        repo_root(), "assets/checkpoints/wav2vec2/wav2vec2_fairseq_base_ls960.pth"
    ),
    "wav2vec": join(repo_root(), "assets/checkpoints/wav2vec/wav2vec_large.pt"),
    "vq_wav2vec": join(repo_root(), "assets/checkpoints/wav2vec/vq-wav2vec.pt"),
    "wavlm_base": join(repo_root(), "assets/checkpoints/wavlm/WavLM-Base.pt"),
    "wavlm_base+": join(repo_root(), "assets/checkpoints/wavlm/WavLM-Base+.pt"),
    "cpc": join(repo_root(), "assets/checkpoints/cpc/60k_epoch4-d0f474de.pt"),
}
NAMES = list(CHECKPOINTS.keys())


# TODO: make Hubert/Wav2Vec2 be able to use causal attention
def load_hubert_base_custom():
    sd = torch.load(CHECKPOINTS["hubert_base"])
    model = hubert_base()
    model.load_state_dict(sd)
    model.sample_rate = HUBERT_BASE.sample_rate
    model.name = "hubert_base"
    return model


def load_wav2vec2_base_custom():
    sd = torch.load(CHECKPOINTS["wav2vec2_base"])
    model = wav2vec2_base()
    model.load_state_dict(sd)
    model.sample_rate = HUBERT_BASE.sample_rate
    model.name = "wav2vec2_base"
    return model


def load_CPC():
    """
    Contrast predictive learning model for audio data
    pretrained: if True, load a model trained on libri-light 60k
    (https://arxiv.org/abs/1912.07875)
    **kwargs : see cpc/cpc_default_config to get the list of possible arguments
    """

    import argparse
    from cpc.model import CPCModel as cpcmodel
    from cpc.cpc_default_config import get_default_cpc_config
    from cpc.feature_loader import getEncoder, getAR, loadArgs

    locArgs = get_default_cpc_config()

    if exists(CHECKPOINTS["cpc"]):
        checkpoint = torch.load(CHECKPOINTS["cpc"], map_location="cpu")
    else:
        checkpoint_url = "https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/60k_epoch4-d0f474de.pt"
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_url, progress=False, map_location="cpu"
        )
        makedirs(dirname(CHECKPOINTS["cpc"]))
        torch.save(checkpoint, CHECKPOINTS["cpc"])
    loadArgs(locArgs, argparse.Namespace(**checkpoint["config"]))
    encoderNet = getEncoder(locArgs)
    arNet = getAR(locArgs)
    model = cpcmodel(encoderNet, arNet)

    # always load pretrained
    model.load_state_dict(checkpoint["weights"], strict=False)
    model.name = "cpc"
    return model


def load_hubert_base():
    model = HUBERT_BASE.get_model()
    model.sample_rate = HUBERT_BASE.sample_rate
    model.name = "hubert_base"
    return model


def load_wav2vec2_base():
    model = WAV2VEC2_BASE.get_model()
    model.sample_rate = HUBERT_BASE.sample_rate
    model.name = "wav2vec2_base"
    return model


def load_wavlm(name="wavlm_base+"):
    """
    Original Repo: https://github.com/microsoft/unilm/tree/master/wavlm
    """

    # load the pre-trained checkpoints
    # load the pre-trained checkpoints
    checkpoint = torch.load(CHECKPOINTS[name])
    cfg = WavLMConfig(checkpoint["cfg"])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.name = name
    return model


def load_wav2vec(name="wav2vec"):
    from fairseq.checkpoint_utils import load_model_ensemble_and_task

    cp_path = CHECKPOINTS[name]
    model, _, _ = load_model_ensemble_and_task([cp_path])
    model = model[0]
    model.eval()
    model.name = name
    return model


def load_vq_wav2vec(name="vq_wav2vec"):
    cp = CHECKPOINTS[name]
    from fairseq.checkpoint_utils import load_model_ensemble_and_task

    model, _, _ = load_model_ensemble_and_task([cp])
    model = model[0]
    model.eval()
    model.name = name
    return model


def load_pretrained_encoder(name="hubert_base"):
    assert name.lower() in CHECKPOINTS.keys()
    if name == "hubert_base":
        return load_hubert_base_custom()
    elif name == "wav2vec2_base":
        return load_wav2vec2_base_custom()
    elif name == "wav2vec":
        return load_wav2vec(name)
    elif name == "vq_wav2vec":
        return load_wav2vec(name)
    elif name.startswith("wavlm"):
        return load_wavlm(name)
    elif name.lower() == "cpc":
        return load_CPC()
    else:
        raise NotImplementedError(f"{name} not implemented. Try: {NAMES}")


def test_cpc():
    from conv_ssl.utils import count_parameters

    model = load_CPC()
    n = count_parameters(model, as_string=True, learnable=False)
    print(f"CPC Model parameters: {n}")

    diter = iter(dm.train_dataloader())

    batch = next(diter)

    x = batch["waveform"]
    c, z, l = model(x, None)


def test_wavlm():
    model = load_wavlm("wavlm_base+")

    # extract the representation of last layer
    wav_input_16khz = torch.randn(1, 16000)
    rep = model.extract_features(wav_input_16khz, output_layer=6)[0]

    print("rep: ", tuple(rep.shape))
