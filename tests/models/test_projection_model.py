import pytest
import torch
from conv_ssl.models import ProjectionModel


@pytest.mark.projection
@pytest.mark.models
def test_projection_model():
    conf = ProjectionModel.load_config()
    conf["tier1"]["dim"] = 64
    conf["tier2"]["dim"] = conf["tier1"]["dim"]
    conf["tier1"]["num_layers"] = 1
    conf["tier1"]["num_heads"] = 1
    conf["tier2"]["num_layers"] = 1
    conf["tier2"]["num_heads"] = 1
    conf["vad_class_prediction"]["regression"] = False
    model = ProjectionModel(conf)

    input_ids = torch.randint(0, conf["quantizer"]["n_codes"], (1, 100))
    vad = torch.randint(0, 2, (1, 100, 2)).float()
    out = model(input_ids, vad=vad)

    assert tuple(out["z"].shape) == (1, 100, conf["tier1"]["dim"])
    assert tuple(out["logits_ar"].shape) == (1, 100, conf["quantizer"]["n_codes"])
    assert tuple(out["logits_vp"].shape) == (
        1,
        100,
        conf["vad_class_prediction"]["n_classes"],
    )
