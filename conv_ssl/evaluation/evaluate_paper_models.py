from os.path import join
from os import makedirs, cpu_count

from conv_ssl.evaluation.evaluation import evaluate
from conv_ssl.evaluation.utils import get_checkpoint, load_paper_versions
from conv_ssl.utils import everything_deterministic

model_ids = {
    "discrete": {
        "0": "1h52tpnn",
        "1": "3fhjobk0",
        "2": "120k8fdv",
        "3": "1vx0omkd",
        "4": "sbzhz86n",
        "5": "1lyezca0",
        "6": "2vtd1u1n",
        "7": "2ldfo4rg",
        "8": "2ca7uxad",
        "9": "2fsy74rf",
        "10": "3ik6jod6",
    },
    "independent": {
        "0": "1t7vvo0c",
        "1": "24bn5wi6",
        "2": "1u7yzji0",
        "3": "s5unjaj7",
        "4": "10krujrj",
        "5": "2rq33fxr",
        "6": "3uqpk8e1",
        "7": "3mpxa1iy",
        "8": "3ulpo767",
        "9": "3d952gec",
        "10": "2651d3ln",
    },
    "independent_baseline": {
        "0": "2mme28tm",
        "1": "qo9mf26t",
        "2": "2rrdm5ma",
        "3": "mrzizwex",
        "4": "1lximsk1",
        "5": "2wyymo7n",
        "6": "1nze8m3l",
        "7": "1cdhj9yo",
        "8": "kamzjel0",
        "9": "15ze1p0y",
        "10": "2mvwxxar",
    },
    "comparative": {
        "0": "2kwhi1zi",
        "1": "2izpsu6r",
        "2": "23mzxhhd",
        "3": "1lvk73tr",
        "4": "11jlsatj",
        "5": "nxvb62j4",
        "6": "1pglrfbn",
        "7": "1z9qyfh6",
        "8": "1kgiwy2m",
        "9": "1eluv8de",
        "10": "2530040o",
    },
}

everything_deterministic()

savepath = "assets/paper_evaluation"
makedirs(savepath, exist_ok=True)

for model_type, ids in model_ids.items():
    model_root = join(savepath, model_type)
    makedirs(model_root, exist_ok=True)
    for kfold, id in ids.items():
        instance_root = join(model_root, f"kfold_{kfold}")
        makedirs(instance_root, exist_ok=True)

        # get checkpoint from wandb-ID
        checkpoint_path = get_checkpoint(run_path=id)
        checkpoint_path = load_paper_versions(checkpoint_path)

        # Threshold and extract score
        metrics, prediction, curves = evaluate(
            checkpoint_path=checkpoint_path,
            savepath=instance_root,
            batch_size=16,
            num_workers=cpu_count(),
        )
