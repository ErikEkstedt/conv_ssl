import pytest
import torch
from os.path import join
from conv_ssl.models import EncoderPretrained
from conv_ssl.utils import repo_root


@pytest.mark.models
@pytest.mark.encoder
@pytest.mark.parametrize(
    "name",
    ["wav2vec", "vq_wav2vec", "hubert_base", "wav2vec2_base", "wavlm_base+"],
)
def test_encoder(name):
    conf_path = join(repo_root(), "conv_ssl/config")
    if name == "hubert_base":
        conf_path = join(conf_path, "ulm_hubert.yaml")
    elif name == "wav2vec":
        conf_path = join(conf_path, "ulm_wav2vec.yaml")
    elif name == "vq_wav2vec":
        conf_path = join(conf_path, "ulm_vq_wav2vec.yaml")
    elif name == "wav2vec2_base":
        conf_path = join(conf_path, "ulm_wav2vec2.yaml")
    elif name == "wavlm_base+":
        conf_path = join(conf_path, "ulm_wavlm.yaml")
    elif name == "cpc":
        conf_path = join(conf_path, "ulm_cpc.yaml")
    else:
        assert False, "{name} does not exist"

    conf = EncoderPretrained.load_config(conf_path)
    conf["encoder"]["output_layer"] = 6
    conf["quantizer"]["vector_path"] = None
    model = EncoderPretrained(conf)

    # extract the representation of last layer
    wav_input_16khz = torch.randn(1, 16000)

    if torch.cuda.is_available():
        wav_input_16khz = wav_input_16khz.to("cuda")
        model.to("cuda")
    z = model.encode(wav_input_16khz)

    # output_shape = (1, 49, 768)
    # if name == "cpc":
    #     output_shape = (1, 100, 256)
    # elif name in ["wav2vec", "vq_wav2vec"]:
    #     output_shape = (1, 98, 256)
    #
    # assert tuple(z.shape) == output_shape, f"{name} output_shape != {output_shape}"
