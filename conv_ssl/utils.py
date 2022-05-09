from omegaconf import OmegaConf
from os.path import dirname
import torch
from torch.nn.utils.rnn import pad_sequence
from hydra import compose, initialize
import json


def everything_deterministic():
    """
    -----------------------------
    Wav2Vec
    -------
    1. Settings
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(mode=True)
    2. Load Model
    3. backprop from step and plot

    RuntimeError: replication_pad1d_backward_cuda does not have a deterministic
    implementation, but you set 'torch.use_deterministic_algorithms(True)'. You can
    turn off determinism just for this operation if that's acceptable for your
    application. You can also file an issue at
    https://github.com/pytorch/pytorch/issues to help us prioritize adding
    deterministic support for this operation.


    -----------------------------
    CPC
    -------
    1. Settings
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(mode=True)
    2. Load Model
    3. backprop from step and plot

    RuntimeError: Deterministic behavior was enabled with either
    `torch.use_deterministic_algorithms(True)` or
    `at::Context::setDeterministicAlgorithms(true)`, but this operation is not
    deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable
    deterministic behavior in this case, you must set an environment variable
    before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or
    CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to
    https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility


    Set these ENV variables and it works with the above recipe

    bash:
        export CUBLAS_WORKSPACE_CONFIG=:4096:8
        export CUBLAS_WORKSPACE_CONFIG=:16:8

    """
    from os import environ

    environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True)


def tensor_dict_to_json(d):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            v = v.tolist()
        elif isinstance(v, dict):
            v = tensor_dict_to_json(v)
        new_d[k] = v
    return new_d


def to_device(batch, device="cuda"):
    new_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            new_batch[k] = v.to(device)
        else:
            new_batch[k] = v
    return new_batch


def repo_root():
    """
    Returns the absolute path to the git repository
    """
    # return (
    #     run(["git", "rev-parse", "--show-toplevel"], capture_output=True)
    #     .stdout.decode()
    #     .strip()
    # )
    root = dirname(__file__)
    root = dirname(root)
    return root


def write_json(data, filename):
    with open(filename, "w", encoding="utf-8") as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False)


def read_json(path, encoding="utf8"):
    with open(path, "r", encoding=encoding) as f:
        data = json.loads(f.read())
    return data


def read_txt(path, encoding="utf-8"):
    data = []
    with open(path, "r", encoding=encoding) as f:
        for line in f.readlines():
            data.append(line.strip())
    return data


def count_parameters(model, as_string=True, learnable=True):
    """
    Count the numper of parameters in a model. If learnable == True only count those that requires gradient else all.

    as_string: Bool, if True provides easily readable string else the value
    """
    n = 0
    for p in model.parameters():
        if learnable:
            if p.requires_grad:
                n += p.nelement()
        else:
            n += p.nelement()

    if as_string:
        m = ""
        for i, j in enumerate(range(len(str(n)) - 1, -1, -1)):
            if i > 0 and i % 3 == 0:
                m += ","
            m += str(n)[j]
        m = m[::-1]
        return m
    return n


def find_island_idx_len(x):
    """
    Finds patches of the same value.

    starts_idx, duration, values = find_island_idx_len(x)

    e.g:
        ends = starts_idx + duration

        s_n = starts_idx[values==n]
        ends_n = s_n + duration[values==n]  # find all patches with N value

    """
    assert x.ndim == 1
    n = len(x)
    y = x[1:] != x[:-1]  # pairwise unequal (string safe)
    i = torch.cat(
        (torch.where(y)[0], torch.tensor(n - 1, device=x.device).unsqueeze(0))
    ).long()
    it = torch.cat((torch.tensor(-1, device=x.device).unsqueeze(0), i))
    dur = it[1:] - it[:-1]
    idx = torch.cumsum(
        torch.cat((torch.tensor([0], device=x.device, dtype=torch.long), dur)), dim=0
    )[
        :-1
    ]  # positions
    return idx, dur, x[i]


def unit_condense_repetition(x):
    """
    Condense repetitions and get the variable length inputs.


    Name: unit_condense_repetition
    Args:
        x:              torch.Tensor: (B, N)

    Repetition:
        condense:       torch.tensor: (B, M)
        duration:       torch.tensor: (B, M)
        lenths:         list: (B,)
    """
    # find_island_idx
    if x.ndim == 1:
        _, duration, condense = find_island_idx_len(x)
        lengths = len(duration)
    else:
        condense, duration, lengths = [], [], []
        for b in range(x.shape[0]):
            _, dur, val = find_island_idx_len(x[b])
            condense.append(val)
            duration.append(dur)
            lengths.append(len(val))
        condense = pad_sequence(condense, batch_first=True)
        duration = pad_sequence(duration, batch_first=True)
    return condense, duration, lengths


def condense_to_original(condense, duration, lengths):
    """
    Condense repetitions and get the variable length inputs.


    Name: unit_condense_repetition
    Args:
        x:              torch.Tensor: (B, N)

    Repetition:
        condense:       torch.tensor: (B, M)
        duration:       torch.tensor: (B, M)
        lenths:         list: (B,)
    """

    if condense.ndim > 1:
        units = []
        for c, d, l in zip(condense, duration, lengths):
            units.append(torch.repeat_interleave(c[:l], d[:l]))
        units = torch.stack(units)
    else:
        units = torch.repeat_interleave(condense, duration)
    return units


def load_config(path=None, args=None, format="dict"):
    conf = OmegaConf.load(path)
    if args is not None:
        conf = OmegaConfArgs.update_conf_with_args(conf, args)

    if format == "dict":
        conf = OmegaConf.to_object(conf)
    return conf


def load_hydra_conf(config_path="conf", config_name="config"):
    """https://stackoverflow.com/a/61169706"""
    try:
        initialize(config_path=config_path)
    except:
        pass

    cfg = compose(config_name=config_name)
    return cfg


class OmegaConfArgs:
    """
    This is annoying... And there is probably a SUPER easy way to do this... But...

    Desiderata:
        * Define the model completely by an OmegaConf (yaml file)
            - OmegaConf argument syntax  ( '+segments.c1=10' )
        * run `sweeps` with WandB
            - requires "normal" argparse arguments (i.e. '--batch_size' etc)

    This class is a helper to define
    - argparse from config (yaml)
    - update config (loaded yaml) with argparse arguments


    See ./config/sosi.yaml for reference yaml
    """

    @staticmethod
    def add_argparse_args(parser, conf, omit_fields=None):
        for field, settings in conf.items():
            if omit_fields is None:
                for setting, value in settings.items():
                    name = f"--{field}.{setting}"
                    parser.add_argument(name, default=None, type=type(value))
            else:
                if not any([field == f for f in omit_fields]):
                    for setting, value in settings.items():
                        name = f"--{field}.{setting}"
                        parser.add_argument(name, default=None, type=type(value))
        return parser

    @staticmethod
    def update_conf_with_args(conf, args, omit_fields=None):
        if not isinstance(args, dict):
            args = vars(args)

        for field, settings in conf.items():
            if omit_fields is None:
                for setting in settings:
                    argname = f"{field}.{setting}"
                    if argname in args and args[argname] is not None:
                        conf[field][setting] = args[argname]
            else:
                if not any([field == f for f in omit_fields]):
                    for setting in settings:
                        argname = f"{field}.{setting}"
                        if argname in args:
                            conf[field][setting] = args[argname]
        return conf
