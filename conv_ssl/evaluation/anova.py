from os.path import join
from glob import glob

import torch
import scipy.stats as stats

from conv_ssl.utils import read_json


METRIC_2_STATS = {
    "f1_hold_shift": "SH",
    "f1_short_long": "SL",
    "f1_predict_shift": "S-pred",
    "f1_bc_prediction": "BC-pred",
}

METRIC_NAMES = [
    "f1_hold_shift",
    "f1_short_long",
    "f1_predict_shift",
    "f1_bc_prediction",
]

STATS_NAMES = ["SH", "SL", "S-pred", "BC-pred"]


def load_all_scores(root="assets/paper_evaluation"):
    """
    load scores create by `evaluate_paper_models.py`

    --------------------------------
    root/
    └── discrete/
        ├── kfold_0
        ├── ...
        └── kfold_11
    └── independent/
        ├── kfold_0
        ├── ...
        └── kfold_11
    └── independent_baseline/
        ├── kfold_0
        ├── ...
        └── kfold_11
    --------------------------------

    """
    all_score = {}
    for model_type in ["discrete", "independent", "independent_baseline"]:
        # model_type_dir = join(root, model_type)
        model_dict = {}
        for metric_filepath in glob(join(root, model_type, "**/metric.json")):
            r = read_json(metric_filepath)
            for tmp_metric, val in r.items():
                if "threshold" in tmp_metric:
                    continue
                if "loss" in tmp_metric:
                    continue
                if isinstance(val, dict):
                    # add shift/hold f1
                    continue
                tmp_stat = METRIC_2_STATS[tmp_metric]
                if tmp_stat in model_dict:
                    model_dict[tmp_stat].append(val)
                else:
                    model_dict[tmp_stat] = [val]
        all_score[model_type] = model_dict
    return all_score


def anova(all_score):
    statistics = {name: {} for name in STATS_NAMES}
    averages = {}
    for stat_name in STATS_NAMES:
        # Get score for the different groups
        m_discrete = all_score["discrete"][stat_name]
        m_ind = all_score["independent"][stat_name]
        m_ind_base = all_score["independent_baseline"][stat_name]

        anova_result = stats.f_oneway(m_discrete, m_ind, m_ind_base)
        statistics[stat_name] = anova_result.pvalue
        averages[stat_name] = {
            "discrete": torch.tensor(m_discrete).mean().item(),
            "independent": torch.tensor(m_ind).mean().item(),
            "independent_baseline": torch.tensor(m_ind_base).mean().item(),
        }

        # ad-hoc test
        d_vs_i = stats.ttest_ind(m_discrete, m_ind)
        d_vs_ib = stats.ttest_ind(m_discrete, m_ind_base)
        statistics[f"{stat_name}_t_test_d_vs_i"] = d_vs_i.pvalue
        statistics[f"{stat_name}_t_test_d_vs_ib"] = d_vs_ib.pvalue

    return statistics


if __name__ == "__main__":

    scores = load_all_scores()
    # for stat, vals in scores["discrete"].items():
    #     vals = torch.tensor(vals)
    #     scores["independent"][stat] = (
    #         vals - 0.02 * torch.rand_like(vals).abs()
    #     ).tolist()
    #     scores["independent_baseline"][stat] = (
    #         vals - 0.04 * torch.rand_like(vals).abs()
    #     ).tolist()
    statistics = anova(scores)  # pvalues

    for k, v in statistics.items():
        print(f"{k}: {v}")
