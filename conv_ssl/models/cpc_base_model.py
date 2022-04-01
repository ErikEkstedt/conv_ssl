import torch

import argparse
from os.path import join, exists, dirname
from os import makedirs

from cpc.model import CPCModel as cpcmodel
from cpc.cpc_default_config import get_default_cpc_config
from cpc.feature_loader import getEncoder, getAR, loadArgs

from conv_ssl.utils import repo_root

"""
torch.hub downloads to: 

    `~/.cache/torch/hub/checkpoints/`

Explicit checkpoint path saved manually in "assets/" see CHECKPOINTS below.
"""


CHECKPOINTS = {
    "cpc": join(repo_root(), "assets/checkpoints/cpc/60k_epoch4-d0f474de.pt")
}
NAMES = list(CHECKPOINTS.keys())


def load_CPC():
    """
    Contrast predictive learning model for audio data
    pretrained: if True, load a model trained on libri-light 60k
    (https://arxiv.org/abs/1912.07875)
    **kwargs : see cpc/cpc_default_config to get the list of possible arguments
    """
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
