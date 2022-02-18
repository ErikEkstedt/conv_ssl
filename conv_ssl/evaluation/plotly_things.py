import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

A_COLOR = ["#0000FF", "#0101A8"]
B_COLOR = ["#FFA500", "#A06700"]
CURRENT_COLOR = "#D42B2B"


def forward(batch, model):
    model.val_metric.reset()
    with torch.no_grad():
        batch = to_device(batch, model.device)
        _, out, batch = model.shared_step(batch)
        out["next_probs"], out["pre_probs"] = model.get_next_speaker_probs(
            out["logits_vp"], vad=batch["vad"]
        )
        out["events"] = model.val_metric.extract_events(batch["vad"])
        model.val_metric.update(
            out["next_probs"], None, events=out["events"], bc_pre_probs=out["pre_probs"]
        )
        out["result"] = model.val_metric.compute()
        if model.conf["vad_projection"]["regression"]:
            out["probs"] = out["logits_vp"].sigmoid()
        else:
            out["probs"] = out["logits_vp"].softmax(dim=-1)
            topk_p, topk_idx = out["probs"].topk(dim=-1, k=5)
            topk_onehot = model.projection_codebook.idx_to_onehot(topk_idx)
            out["topk"] = {"p": topk_p, "idx": topk_idx, "onehot": topk_onehot}
    return out, batch


##########################################################
# Static
##########################################################
def get_vad_traces(vad):
    n_frames = vad.shape[0]
    x = np.arange(0, n_frames)
    vad_traces = [
        go.Scattergl(
            visible=True,
            line=dict(width=0, color=A_COLOR[0]),
            name="A",
            fill="tozeroy",
            x=x,
            y=vad[:, 0],
        ),
        go.Scattergl(
            visible=True,
            line=dict(width=0, color=B_COLOR[0]),
            name="B",
            fill="tozeroy",
            x=x,
            y=-vad[:, 1],
        ),
    ]
    return vad_traces


def get_next_prob_traces(next_probs, pre_probs=None):
    n_frames = next_probs.shape[0]
    x = np.arange(0, n_frames)
    traces = [
        go.Scattergl(
            visible=True,
            line=dict(width=2, color=A_COLOR[0]),
            mode="lines",
            name="A next",
            x=x,
            y=next_probs[:, 0],
        ),
        go.Scattergl(
            visible=True,
            mode="lines",
            line=dict(width=2, color=B_COLOR[0]),
            name="B next",
            x=x,
            y=next_probs[:, 1] - 1,
        ),
        # 50% cutoff lines
        go.Scattergl(
            visible=True,
            mode="lines",
            line=dict(width=1, color="#444", dash="dash"),
            showlegend=False,
            x=[0, n_frames, None, 0, n_frames],
            y=[0.5, 0.5, None, -0.5, -0.5],
        ),
    ]

    if pre_probs is not None:
        traces.append(
            go.Scattergl(
                visible=True,
                line=dict(width=1, color=A_COLOR[1]),
                name="A pre",
                x=x,
                y=pre_probs[:, 0],
            )
        )

        traces.append(
            go.Scattergl(
                visible=True,
                line=dict(width=1, color=B_COLOR[1]),
                name="B pre",
                x=x,
                y=pre_probs[:, 1] - 1,
            )
        )

    return traces


##########################################################
# Dynamic
##########################################################
def get_independent_step_traces(probs, step_size, bin_times, vad_hz):
    n_frames = probs.shape[0]

    # We create bins for the projection lines
    # We expand the probs/bins to include the current moment (for line plots)
    # bins are extended with 0 (i.e. place on current step)
    # probs are exended with the first value (to get a horizontal line over the first bin)
    bins = np.array([0] + bin_times) * vad_hz
    bins = bins.cumsum()
    probs = np.concatenate((probs[..., :1], probs), axis=-1)

    traces = []
    for step in np.arange(0, n_frames, step_size):
        am = round(probs[step, 0].sum() / len(bin_times), 2)
        bm = round(probs[step, 1].sum() / len(bin_times), 2)
        traces.append(
            go.Scattergl(
                visible=False,
                line=dict(color=CURRENT_COLOR, width=2),
                name="T = " + str(step),
                x=[step, step],
                y=[-1, 1],
            )
        )
        traces.append(
            go.Scattergl(
                visible=False,
                line=dict(shape="vh", color=A_COLOR[0], width=4),
                name="A proj: " + str(am),
                x=step + bins,
                y=probs[step, 0],
            )
        )
        traces.append(
            go.Scattergl(
                visible=False,
                line=dict(shape="vh", color=B_COLOR[0], width=4),
                name="B proj: " + str(bm),
                x=step + bins,
                y=probs[step, 1] - 1,
            )
        )

    # We add 3 traces in each step
    chunk_size = 3
    return traces, chunk_size


def get_projection_window_static_traces(k, bin_times, vad_hz):
    """
    hello
    """

    boundary_width = 2
    # prepare bin_frames
    kmax = 1
    bins = np.array([0] + bin_times) * vad_hz
    bins = bins.cumsum()
    n_hor_lines = 2 * k + 1
    kmin = kmax + 1 - n_hor_lines
    ys = np.arange(kmin, kmax + 1)

    box_traces = []
    # Box lines for projection window
    # add the frame of the box. Start in lowest corner -> right -> up -> left -> down
    x0, x1 = bins[0], bins[-1]
    y0, y1 = ys[0], ys[-1]
    x = [x0, x1, x1, x0, x0]
    y = [y0, y0, y1, y1, y0]
    # Add bin-delimiters
    for b in bins[1:-1]:
        x += [None, b, b]  # None makes it "jump"=line is invisible
        y += [None, y0, y1]
    # Add horizontal lines
    for yy in ys[::2][1:-1]:
        y += [None, yy, yy]
        x += [None, x0, x1]
    box_traces.append(
        go.Scattergl(
            visible=True,
            showlegend=False,
            mode="lines",
            line=dict(color="#222", width=boundary_width),
            x=x,
            y=y,
        )
    )

    # Add horizontal lines
    y = []
    x = []
    for y_tmp in ys[1::2]:
        x += [x0, x1, None]
        y += [y_tmp, y_tmp, None]
    x = x[:-1]
    y = y[:-1]
    box_traces.append(
        go.Scattergl(
            visible=True,
            showlegend=False,
            mode="lines",
            line=dict(color="#555", width=1),
            x=x,
            y=y,
        )
    )
    return box_traces


def get_topk_traces(probs, onehot, bin_times, vad_hz, step_size=10):
    n_frames = probs.shape[0]
    k = probs.shape[-1]
    x = np.array(bin_times) * vad_hz
    x = x.cumsum()
    step_traces = []
    current_steps = []
    for step in range(0, n_frames, step_size):
        current_steps.append(
            go.Scattergl(
                visible=False,
                mode="lines",
                line=dict(width=2, color=CURRENT_COLOR),
                showlegend=False,
                x=[step, step],
                y=[-1, 1],
            )
        )
        k_traces = []
        for nk in range(k):
            oh = onehot[step, nk]
            p = probs[step, nk] * 2  # scale to cover entire y
            tmp_k = []
            # Probability
            tmp_k.append(
                go.Scattergl(
                    visible=False,
                    mode="lines",
                    line=dict(shape="vh", width=20, color="red"),
                    showlegend=False,
                    x=[-11, -11],
                    y=[-1, p - 1],
                )
            )

            # Channel 0 area fill
            tmp_k.append(
                go.Scattergl(
                    visible=False,
                    mode="lines",
                    line=dict(shape="vh", width=0, color=A_COLOR[0]),
                    showlegend=False,
                    fill="tozeroy",
                    x=x,
                    y=oh[0],
                )
            )
            # Channel 1 area fill
            tmp_k.append(
                go.Scattergl(
                    visible=False,
                    mode="lines",
                    fill="tozeroy",
                    line=dict(shape="vh", width=0, color=B_COLOR[0]),
                    showlegend=False,
                    x=x,
                    y=-oh[1],
                )
            )
            k_traces.append(tmp_k)
        step_traces.append(k_traces)
    return step_traces, current_steps


def get_steps(n_traces, n_always_visible, chunk_size):
    steps = []
    n_steps = n_traces // chunk_size
    for i in range(n_steps):
        step = {
            "method": "update",  # update the traces
            "args": [{"visible": [True] * n_always_visible + [False] * n_traces}],
        }
        start_trace_idx = chunk_size * i
        offset = n_always_visible + start_trace_idx  # offset with unchanging traces
        for n in range(chunk_size):
            step["args"][0]["visible"][offset + n] = True
        steps.append(step)
    return steps


##########################################################
# Full
##########################################################
def plotly_independent(
    probs,
    next_probs,
    pre_probs,
    vad,
    bin_times=[0.2, 0.4, 0.6, 0.8],
    vad_hz=100,
    step_size=10,
    figsize=(900, 600),
):
    fig = go.Figure(layout=dict(width=figsize[0], height=figsize[1]))
    ######################################################################
    # Static
    vad_traces = get_vad_traces(vad)
    next_prob_traces = get_next_prob_traces(next_probs, pre_probs)
    ######################################################################
    # Changed on step
    ind_traces, chunk_size = get_independent_step_traces(
        probs, step_size, bin_times, vad_hz
    )
    ######################################################################
    # Add all the traces to the figure
    fig.add_traces(vad_traces)
    fig.add_traces(next_prob_traces)
    fig.add_traces(ind_traces)
    ######################################################################
    # Create step information (toggles `visible` appropriatly) for slider
    n_always_visible = len(vad_traces) + len(next_prob_traces)
    n_traces = n_always_visible + len(ind_traces)
    steps = get_steps(n_traces, n_always_visible, chunk_size=chunk_size)
    slider = {
        "active": 0,
        "steps": steps,
    }
    ######################################################################
    # Add slider and axis info
    fig.update_layout(
        sliders=[slider],
        yaxis=dict(
            autorange=False, range=[-1.05, 1.05], zeroline=True, zerolinecolor="#000"
        ),
        xaxis=dict(autorange=False, range=[0, vad.shape[0]]),
    )
    return fig


def plotly_discrete(
    probs,
    onehot,
    next_probs,
    vad,
    bin_times=[0.2, 0.4, 0.6, 0.8],
    vad_hz=100,
    step_size=10,
    figsize=(1200, 600),
):
    """
    bin_times=[0.2, 0.4, 0.6, 0.8]
    vad_hz=100
    step_size=10
    figsize=(1200, 600)
    """

    specs = [
        [{"rowspan": 5, "type": "scatter"}, {"type": "scatter"}],
        [None, {"type": "scatter"}],
        [None, {"type": "scatter"}],
        [None, {"type": "scatter"}],
        [None, {"type": "scatter"}],
    ]
    #######################################################################
    fig = go.Figure(layout=dict(width=figsize[0], height=figsize[1]))
    fig = make_subplots(
        rows=5,
        cols=2,
        column_widths=[3, 1],
        specs=specs,
        horizontal_spacing=0.005,
        figure=fig,
    )
    #######################################################################
    # Static Figure
    n_always_visible = 0
    n_traces = 0
    vad_traces = get_vad_traces(vad)
    next_prob_traces = get_next_prob_traces(next_probs)
    fig.add_traces(vad_traces, rows=1, cols=1)
    fig.add_traces(next_prob_traces, rows=1, cols=1)
    n_always_visible += len(vad_traces)
    n_always_visible += len(next_prob_traces)
    n_traces += len(vad_traces)
    n_traces += len(next_prob_traces)
    # draw box + bin-delimiters
    win_proj_box_traces = get_projection_window_static_traces(
        k=1, bin_times=bin_times, vad_hz=vad_hz
    )
    for row in range(1, 5 + 1):
        fig.add_traces(win_proj_box_traces, rows=row, cols=2)
    n_always_visible += len(win_proj_box_traces) * 5
    n_traces += len(win_proj_box_traces) * 5
    fig.update_layout(
        yaxis2=dict(showticklabels=False, showgrid=False),
        yaxis3=dict(showticklabels=False, showgrid=False),
        yaxis4=dict(showticklabels=False, showgrid=False),
        yaxis5=dict(showticklabels=False, showgrid=False),
        yaxis6=dict(showticklabels=False, showgrid=False),
        xaxis2=dict(showticklabels=False, showgrid=False),
        xaxis3=dict(showticklabels=False, showgrid=False),
        xaxis4=dict(showticklabels=False, showgrid=False),
        xaxis5=dict(showticklabels=False, showgrid=False),
        xaxis6=dict(showticklabels=False, showgrid=False),
    )
    ##############################
    topk_traces, current_traces = get_topk_traces(
        probs, onehot, bin_times, vad_hz, step_size
    )
    n_traces += len(current_traces)
    chunk_size = (len(topk_traces[0][0]) * 5) + 1
    for cur_trace, k_traces in zip(current_traces, topk_traces):
        fig.add_trace(cur_trace, row=1, col=1)
        for nk, ktrace in enumerate(k_traces):
            row = nk + 1
            fig.add_traces(ktrace, rows=row, cols=2)
            n_traces += len(ktrace)
    steps = get_steps(n_traces, n_always_visible, chunk_size)
    slider = {
        "active": 0,
        "steps": steps,
        "pad": {"l": 0, "r": 0},
        "len": 0.75,
    }
    ######################################################################
    # Add slider and axis info
    fig.update_layout(sliders=[slider])
    return fig


if __name__ == "__main__":
    import torch
    from conv_ssl.evaluation.utils import load_model, load_dm
    from conv_ssl.utils import to_device

    # run_path = "how_so/VPModel/27ly86w3"  # independent (same bin size)
    run_path = "how_so/VPModel/2wbyll6r"  # discrete
    model = load_model(run_path=run_path)
    dm = load_dm(model, batch_size=4, num_workers=0)

    batch = next(iter(dm.val_dataloader()))
    out, batch = forward(batch, model)
    b = 1
    probs = out["probs"][b].cpu()
    next_probs = out["next_probs"][b].cpu()
    pre_probs = None
    if out["pre_probs"] is not None:
        pre_probs = out["pre_probs"][b].cpu()
        print("pre_probs: ", tuple(pre_probs.shape))
    vad = batch["vad"][b].cpu()
    print("probs: ", tuple(probs.shape))
    print("next_probs: ", tuple(next_probs.shape))
    print("vad: ", tuple(vad.shape))
    if "topk" in out:
        topk = {
            "p": out["topk"]["p"][b].cpu(),
            "idx": out["topk"]["idx"][b].cpu(),
            "onehot": out["topk"]["onehot"][b].cpu(),
        }
        print("topk p: ", tuple(topk["p"].shape))
        print("topk oh: ", tuple(topk["onehot"].shape))

    # Independent FIG
    # fig = plotly_independent(probs, next_probs, pre_probs, vad)
    # fig.show()

    # Discrete FIG
    probs = topk["p"].numpy()
    onehot = topk["onehot"].numpy()
    idx = topk["idx"].numpy()
    fig = plotly_discrete(probs=probs, onehot=onehot, vad=vad)
    fig.show()
