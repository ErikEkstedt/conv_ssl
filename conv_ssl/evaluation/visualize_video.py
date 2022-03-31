import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import subprocess
import torch
import torchaudio

from conv_ssl.utils import to_device
from conv_ssl.evaluation.utils import load_model, load_dm
from conv_ssl.plot_utils import plot_vad_label, plot_vad_oh, plot_batch

from datasets_turntaking import DialogAudioDM


# load data
# load model
# batch forward pass
# tmp waveform save
# p, p_bc

event_kwargs = dict(
    shift_onset_cond=1,
    shift_offset_cond=1,
    hold_onset_cond=1,
    hold_offset_cond=1,
    min_silence=0.15,
    non_shift_horizon=2.0,
    non_shift_majority_ratio=0.95,
    metric_pad=0.05,
    metric_dur=0.1,
    metric_onset_dur=0.2,
    metric_pre_label_dur=0.5,
    metric_min_context=1.0,
    min_context=3.0,
    bc_max_duration=1.0,
    bc_pre_silence=1.0,
    bc_post_silence=2.0,
)


def join_batch_data(
    probs, batch, logits, overlap_frames, overlap_samples, max_batch=9999
):
    """
    # Join batch together
    """
    logs = [logits[0]]
    p = [probs["p"][0]]
    p_bc = [probs["bc_prediction"][0]]

    x = [batch["waveform"][0]]
    va = [batch["vad"][0]]
    ev = {k: [v[0]] for k, v in events.items()}
    for i in range(1, min(max_batch, batch["vad"].shape[0])):
        logs.append(logits[i, overlap_frames:])
        p.append(probs["p"][i, overlap_frames:])
        p_bc.append(probs["bc_prediction"][i, overlap_frames:])
        x.append(batch["waveform"][i, overlap_samples:])
        va.append(batch["vad"][i, overlap_frames:])
        {ev[k].append(v[0]) for k, v in events.items()}
    # return {
    #     "va": torch.cat(va).cpu(),
    #     "x": torch.cat(x).cpu(),
    #     "p": torch.cat(p).cpu(),
    #     "p_bc": torch.cat(p_bc).cpu(),
    #     "ev": {k: torch.cat(v).cpu() for k, v in ev.items()},
    # }
    ev = {k: torch.cat(v).cpu() for k, v in ev.items()}
    return (
        torch.cat(logs).cpu(),
        torch.cat(p).cpu(),
        torch.cat(p_bc).cpu(),
        torch.cat(x).cpu(),
        torch.cat(va).cpu(),
        ev,
    )


def get_weighted_onehot(logits, model):
    probs = logits.softmax(-1)
    ohs = model.projection_codebook.idx_to_onehot(
        torch.arange(model.projection_codebook.n_classes).cuda()
    ).cpu()
    ohs = probs.unsqueeze(-1).unsqueeze(-1) * ohs
    ohs = ohs.sum(dim=1)  # sum all class onehot
    return ohs


class VAPanimation:
    def __init__(
        self,
        logits,
        p,
        p_bc,
        x,
        va,
        ev,
        window_duration=10,
        frame_hz=100,
        sample_rate=16000,
        fps=20,
        dpi=200,
        bin_frames=[20, 40, 60, 80],
    ) -> None:
        """"""
        # Model output
        self.logits = logits
        self.p = p
        self.p_bc = p_bc

        self.weighted_oh = get_weighted_onehot(logits, model)
        self.best_idx = logits.max(dim=-1).indices
        self.best_p = logits.softmax(-1).max(dim=-1).values[self.best_idx]

        # Model input
        self.x = x
        self.va = va

        # Events
        self.ev = ev

        # Parameters
        self.frame_hz = frame_hz
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.window_frames = self.window_duration * self.frame_hz
        self.center = self.window_frames // 2
        self.bin_frames = bin_frames

        # Animation params
        self.dpi = dpi
        self.fps = fps
        self.frame_step = int(100.0 / self.fps)

        self.plot_kwargs = {
            "A": {"color": "b"},
            "B": {"color": "orange"},
            "va": {"alpha": 0.6},
            "bc": {"color": "darkgreen", "alpha": 0.6},
            "vap": {"ylim": [-0.5, 0.5], "width": 3},
            "current": {"width": 5},
        }

        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 4))
        self.pred_ax = self.ax.twinx()
        self.vap_ax = self.ax.twinx()
        self.vap_patches = []
        self.draw_vap_patches = True

        self.draw_static()
        self.started = False

    def set_axis_lim(self):
        self.ax.set_xlim([0, self.window_frames])
        # PROBS
        self.pred_ax.set_xlim([0, self.window_frames])
        self.pred_ax.set_yticks([])
        # VAP
        self.vap_ax.set_ylim([-1, 1])
        self.vap_ax.set_yticks([])

    def draw_static(self):
        self.current_line = self.pred_ax.vlines(
            self.center,
            ymin=-1,
            ymax=1,
            color="r",
            linewidth=self.plot_kwargs["current"]["width"],
        )

        # VAP BOX
        s = (torch.tensor(self.bin_frames).cumsum(0) + self.center).tolist()
        ymin, ymax = self.plot_kwargs["vap"]["ylim"]
        w = s[-1] - self.center
        h = ymax - ymin
        # white background

        vap_background = Rectangle(
            xy=[self.center, ymin], width=w, height=h, color="w", alpha=1
        )
        self.vap_ax.add_patch(vap_background)
        self.vap_ax.vlines(
            s,
            ymin=ymin,
            ymax=ymax,
            color="k",
            linewidth=self.plot_kwargs["vap"]["width"],
        )
        self.vap_ax.plot(
            [self.center, s[-1]],
            [ymin, ymin],
            color="k",
            linewidth=self.plot_kwargs["vap"]["width"],
        )
        self.vap_ax.plot(
            [self.center, s[-1]],
            [ymax, ymax],
            color="k",
            linewidth=self.plot_kwargs["vap"]["width"],
        )

    def clear_ax(self):
        self.ax.cla()
        # self.pred_ax.cla()

        self.pa.remove()
        self.pb.remove()
        self.p_bc_a.remove()
        self.p_bc_b.remove()

        # self.vap_ax.patches = [self.vap_ax.patches[0]]
        for i in range(len(self.vap_patches)):
            self.vap_patches[i].remove()
        # self.vap_ax.patches.clear()

    def draw_step(self, step=0):
        if not self.started:
            self.started = True
        else:
            self.clear_ax()

        end = step + self.window_frames

        _ = plot_vad_oh(
            self.va[step:end], ax=self.ax, alpha=self.plot_kwargs["va"]["alpha"]
        )

        # Draw probalitiy curves
        (self.pa,) = self.pred_ax.plot(
            self.p[step:end, 0], color=self.plot_kwargs["A"]["color"]
        )
        (self.p_bc_a,) = self.pred_ax.plot(
            self.p_bc[step:end, 0], color=self.plot_kwargs["bc"]["color"]
        )
        (self.pb,) = self.pred_ax.plot(
            self.p[step:, 1] - 1, color=self.plot_kwargs["B"]["color"]
        )
        (self.p_bc_b,) = self.pred_ax.plot(
            self.p_bc[step:end, 1] - 1, color=self.plot_kwargs["bc"]["color"]
        )

        # draw weighted oh projection

        h = self.plot_kwargs["vap"]["ylim"][-1]

        jj = 0
        for speaker, sp_color in zip(
            [0, 1], [self.plot_kwargs["A"]["color"], self.plot_kwargs["B"]["color"]]
        ):
            bf_cum = 0
            for bin, bf in enumerate(self.bin_frames):

                alpha = self.weighted_oh[step + self.center, speaker, bin].item()

                if self.draw_vap_patches:
                    start = self.center + bf_cum
                    vap_patch = Rectangle(
                        xy=[start, -h * speaker],
                        width=bf,
                        height=h,
                        color=sp_color,
                        alpha=alpha,
                    )
                    # self.vap_patches.append(vap_patch)
                    self.vap_ax.add_patch(vap_patch)
                else:
                    self.vap_ax.patches[jj + 1].set_alpha(alpha)
                # self.vap_patches.append(p)
                bf_cum += bf
                jj += 1

        self.draw_vap_patches = False

    def update(self, step):
        self.draw_step(step)
        self.set_axis_lim()
        return []

    def ffmpeg_call(self, out_path, vid_path, wav_path):
        """
        Overlay the static image on top of the video (saved with transparency) and
        adding the audio.

        Arguments:
            vid_path:  path to temporary dynamic video file
            wav_path:  path to temporary audio file
            img_path:  path to temporary static image
            out_path:  path to save final video to
        """
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            vid_path,
            "-i",
            wav_path,
            "-vcodec",
            "libopenh264",
            out_path,
        ]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        p.communicate()

    def save_video(self, path="test.mp4"):
        tmp_video_path = "/tmp/vap_video_ani.mp4"
        tmp_wav_path = "/tmp/vap_video_audio.wav"
        n_frames = self.p.shape[0] - self.center

        sample_offset = int(self.sample_rate * self.center / self.frame_hz)

        # SAVE tmp waveform
        torchaudio.save(
            tmp_wav_path,
            self.x[sample_offset:].unsqueeze(0),
            sample_rate=self.sample_rate,
        )

        # Save matplot video
        moviewriter = animation.FFMpegWriter(
            fps=self.fps  # , codec="libopenh264", extra_args=["-threads", "16"]
        )
        with moviewriter.saving(ani.fig, tmp_video_path, dpi=self.dpi):
            for step in tqdm(range(0, n_frames, self.frame_step)):
                _ = ani.update(step)
                moviewriter.grab_frame()

        self.ffmpeg_call(path, tmp_video_path, tmp_wav_path)


@torch.no_grad()
def model_forward(batch, model):
    batch = to_device(batch, model.device)
    events = model.test_metric.extract_events(batch["vad"])
    _, out, batch = model.shared_step(batch)
    probs = model.get_next_speaker_probs(out["logits_vp"], batch["vad"])
    for k, v in events.items():
        events[k] = v[:, :1000]
    # logits, p, p_bc, x, va, ev = join_batch_data(
    #     probs, batch, out["logits_vp"], overlap_frames, overlap_samples
    # )
    return join_batch_data(
        probs, batch, out["logits_vp"], overlap_frames, overlap_samples
    )


def dialog_dataset(dset):
    last_session = ""
    current_batch = {
        "waveform": [],
        "vad": [],
        "vad_history": [],
        "session": [],
    }
    for ii, batch in enumerate(dset):
        if ii > 0 and last_session != batch["session"]:
            # yield dialog batch
            for k, v in current_batch.items():
                if k != "session":
                    current_batch[k] = torch.cat(v)
            yield current_batch
            current_batch = {
                "waveform": [],
                "vad": [],
                "vad_history": [],
                "session": [],
            }
            last_session = batch["session"]
            for k, v in batch.items():
                if k == "session":
                    current_batch[k] += [v]
                elif k != "dataset_name":
                    current_batch[k].append(v)
        else:
            last_session = batch["session"]
            for k, v in batch.items():
                if k == "session":
                    current_batch[k] += [v]
                elif k != "dataset_name":
                    current_batch[k].append(v)

    for k, v in current_batch.items():
        if k != "session":
            current_batch[k] = torch.cat(v)

    return current_batch


def dialog_videos():
    # discrete kfold 0
    model = load_model(run_path="1h52tpnn", eval=True, strict=False)
    model = model.eval()
    model.test_metric = model.init_metric(
        model.conf, model.frame_hz, bc_pred_pr_curve=False, **event_kwargs
    )
    model.test_metric = model.test_metric.to(model.device)
    model = model.eval()

    data_conf = DialogAudioDM.load_config()
    horizon = round(sum(model.conf["vad_projection"]["bin_times"]), 2)
    vad_hz = model.frame_hz
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        audio_duration=10,
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=5,
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hz=vad_hz,
        vad_horizon=horizon,
        vad_history=True,
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        flip_channels=False,  # don't flip on evaluation
        batch_size=16,
    )
    dm.prepare_data()
    dm.setup("test")

    dset = dm.test_dset
    # print("d['waveform']: ", tuple(d["waveform"].shape))
    # print("d['vad']: ", tuple(d["vad"].shape))
    # print("d['vad_history']: ", tuple(d["vad_history"].shape))
    # print("d['vad]: ", tuple(d["vad"].shape))

    for batch in dialog_dataset(dset):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {tuple(v.shape)}")
            else:
                print(f"{k}: {v}")
        logits, p, p_bc, x, va, ev = model_forward(batch, model)

        ani = VAPanimation(logits, p, p_bc, x, va, ev, fps=20)
        ani.save_video(f"{batch['session']}.mp4")
        print("saved: ", batch["session"][0])


if __name__ == "__main__":

    # discrete kfold 0
    model = load_model(run_path="1h52tpnn", eval=True, strict=False)
    model = model.eval()
    model.test_metric = model.init_metric(
        model.conf, model.frame_hz, bc_pred_pr_curve=False, **event_kwargs
    )
    model.test_metric = model.test_metric.to(model.device)

    # Model (predictor) has a constrained context size (1024) and so
    # we "patch" together overlapping dialog chunks
    overlap = 5
    overlap_frames = overlap * model.frame_hz
    overlap_samples = overlap * 16000
    # GET A BATCH
    dm = load_dm(batch_size=20, audio_duration=10, audio_overlap=overlap)
    diter = iter(dm.test_dataloader())
    batch = next(diter)
    batch = next(diter)
    batch = next(diter)
    batch = next(diter)
    batch = to_device(batch)

    # FORWARD
    with torch.no_grad():
        events = model.test_metric.extract_events(batch["vad"])
        _, out, batch = model.shared_step(batch)
        probs = model.get_next_speaker_probs(out["logits_vp"], batch["vad"])
        for k, v in events.items():
            events[k] = v[:, :1000]
    logits, p, p_bc, x, va, ev = join_batch_data(
        probs, batch, out["logits_vp"], overlap_frames, overlap_samples
    )
    print("p: ", tuple(p.shape))

    # plt.close("all")
    # ani = VAPanimation(logits, p, p_bc, x, va, ev)
    # ani.draw_step(500)
    # ani.set_axis_lim()
    # plt.pause(0.1)
    ani = VAPanimation(logits, p, p_bc, x, va, ev, fps=20)
    ani.save_video("test.mp4")
    # for step in range(0, ani.p.shape[0] - 1000, 20):
    #     # ani.draw_step(step)
    #     # ani.set_axis_lim()
    #     ani.update(step)
    #     plt.pause(0.2)
    ###########################################
    # fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    # _ = plot_vad_oh(va, ax=ax, alpha=0.6)
    # pred_ax = ax.twinx()
    # pred_ax.plot(p[:, 0], color=plot_kwargs["A"]["color"])
    # pred_ax.plot(p[:, 1], color=plot_kwargs["B"]["color"])
    # plt.pause(0.1)
    # fig, ax = plot_batch(probs, events, batch["vad"], metric="predict_bc")
    # fig, ax = plot_batch(probs, events, batch["vad"], metric="predict_shift")
    # fig, ax = plot_batch(probs, events, batch["vad"], metric="long")
