from os.path import join
import torch

from conv_ssl.evaluation.visualize_metrics import test_single_model, event_kwargs

discrete = {
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
}

independent = {
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
}

independent_baseline = {
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
}

comparative = {
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
}


def test_models(
    id_dict,
    event_kwargs,
    threshold_pred_shift=0.5,
    threshold_short_long=0.5,
    threshold_bc_pred=0.5,
    batch_size=16,
    project_id="how_so/VPModel",
):

    all_data = {}
    all_result = {}
    for kfold, id in id_dict.items():
        if id is None:
            continue
        run_path = join(project_id, id)
        print(f"{kfold} run_path: ", run_path)
        result, _ = test_single_model(
            run_path,
            event_kwargs=event_kwargs,
            threshold_pred_shift=threshold_pred_shift,
            threshold_short_long=threshold_short_long,
            threshold_bc_pred=threshold_bc_pred,
            bc_pred_pr_curve=False,
            shift_pred_pr_curve=False,
            long_short_pr_curve=False,
            batch_size=batch_size,
        )
        all_data[kfold] = result
        # add results
        for metric, value in result[0].items():
            if metric not in all_result:
                all_result[metric] = []
            all_result[metric].append(value)
    return all_result, all_data


if __name__ == "__main__":
    all_result, all_data = test_models(
        discrete,
        event_kwargs=event_kwargs,
        threshold_short_long=0.213,
        threshold_pred_shift=0.018,
        threshold_bc_pred=0.01,
        project_id="how_so/VPModel",
    )
    torch.save(all_result, "all_res_discrete_new.pt")
    all_result, all_data = test_models(
        independent,
        event_kwargs=event_kwargs,
        threshold_short_long=0.071,
        threshold_pred_shift=0.08,
        threshold_bc_pred=0.01,
        project_id="how_so/VPModel",
    )
    torch.save(all_result, "all_res_independent.pt")
    # all_result, all_data = test_models(
    #     independent_baseline, event_kwargs=event_kwargs, project_id="how_so/VPModel"
    # )
    # torch.save(all_result, "all_result_ind_base_new.pt")
    # torch.save(all_data, "all_data_ind_base_new.pt")
    # all_result, all_data = test_models(
    #     comparative, event_kwargs=event_kwargs, project_id="how_so/VPModel"
    # )
    # torch.save(all_result, "all_result_comp_new.pt")
    # torch.save(all_data, "all_data_comp_new.pt")
