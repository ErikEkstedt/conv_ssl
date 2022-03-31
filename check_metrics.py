import torch

from os.path import join
from glob import glob


if __name__ == "__main__":

    root = "/home/erik/projects/conv_ssl/assets/score"
    all_result = {}
    for model in ["discrete", "independent", "independent_baseline", "comparative"]:
        result = {}
        path = join(root, model)
        result_paths = glob(join(path, "*_result.pt"))
        for path in result_paths:
            res = torch.load(path)
            for k, v in res.items():
                if k not in result:
                    result[k] = [v]
                else:
                    result[k].append(v)
        all_result[model] = result

    m = "test/shift_f1"
    stats = {}
    for model, results in all_result.items():
        stats[model] = torch.tensor(results[m]).mean()

    # change name for comp
    root = "/home/erik/projects/conv_ssl/assets/score/comp"
    ps = glob(join(root, "score*.pt"))
    ps.sort()
    new_root = "assets/score/comparative"
    for p in ps:
        n = int(p.split("_")[-1].replace(".pt", ""))
        res = torch.load(p)
        torch.save(res, join(new_root, f"kfold{n}_comparative_result.pt"))
