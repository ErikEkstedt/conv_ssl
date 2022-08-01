import pytest
import torch
from conv_ssl.models import AR
from conv_ssl.utils import load_hydra_conf


@pytest.mark.ar
@pytest.mark.parametrize(
    "config_name", ["model/discrete", "model/discrete_20hz", "model/discrete_50hz"]
)
def test_autoregressive(config_name):

    D = 256
    conf = load_hydra_conf(config_name=config_name)["model"]
    model = AR(
        input_dim=D,
        dim=conf["ar"]["dim"],
        num_layers=conf["ar"]["num_layers"],
        dropout=conf["ar"]["dropout"],
        ar=conf["ar"]["type"],
        transfomer_kwargs=dict(
            num_heads=conf["ar"]["num_heads"],
            dff_k=conf["ar"]["dff_k"],
            use_pos_emb=conf["ar"]["use_pos_emb"],
            max_context=conf["ar"].get("max_context", None),
            abspos=conf["ar"].get("abspos", None),
            sizeSeq=conf["ar"].get("sizeSeq", None),
        ),
    )

    in_frames = 100
    if config_name.endswith("20hz"):
        in_frames = 20
    elif config_name.endswith("50hz"):
        in_frames = 50

    # extract the representation of last layer
    x = torch.randn(1, in_frames, D)

    if torch.cuda.is_available():
        x = x.to("cuda")
        model.to("cuda")

    z = model(x)["z"]
    assert tuple(z.shape) == (1, in_frames, D), "shape mismatch"
