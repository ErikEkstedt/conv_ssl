import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

mpl.use("Agg")

from tqdm import tqdm
import subprocess
from os.path import dirname
from os import remove
from pathlib import Path
import torchaudio

from datasets_turntaking.features.plot_utils import plot_melspectrogram, plot_vad_oh


class VadPredAnimator:
    def __init__(
        self,
        waveform,
        vad,
        vad_label_oh,
        bin_sizes,
        sample_rate=16000,
        figsize=(12, 4),
        fps=10,
        frame_step=None,
        dpi=100,
    ):
        self.waveform = waveform
        self.vad = vad
        self.vad_label_oh = vad_label_oh
        self.bin_sizes = bin_sizes
        self.n_bins = len(bin_sizes)

        self.fps = fps
        if frame_step is not None:
            self.frame_step = frame_step
        else:
            self.frame_step = int(100 / fps)  # Assumed 100hz frame vad
        self.frames = range(0, vad.shape[0], self.frame_step)

        # Data settings
        self.sample_rate = sample_rate

        # Animation
        self.figsize = figsize
        self.dpi = dpi
        self.layout = dict(
            left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.02, hspace=0.02
        )

    def _fix_layout(self):
        plt.tight_layout()
        plt.subplots_adjust(**self.layout)

    def plot_vad_sample_animation(self, waveform, vad, plot=False):
        fig, ax = plt.subplots(4, 1, figsize=self.figsize, dpi=self.dpi)

        assert len(ax) >= 4, "Must provide at least 4 ax"

        _ = plot_melspectrogram(
            waveform,
            ax=ax[0],
            n_mels=80,
            frame_time=0.05,
            hop_time=0.01,
            sample_rate=self.sample_rate,
        )
        ax[0].set_yticks([])
        ax[0].set_xticks([])
        ax[0].set_ylabel("mel")

        w = waveform[::20]
        ax[1].plot(w)
        ax[1].set_xlim([0, len(w)])
        ax[1].set_ylim([-1.0, 1.0])
        ax[1].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_ylabel("waveform")

        _ = plot_vad_oh(
            vad, ax=ax[2], label=["A", "B"], legend_loc="upper right", plot=False
        )
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        ax[2].set_ylabel("vad")

        _ = plot_vad_oh(vad, ax=ax[3], alpha=0.1, plot=False)
        ax[3].set_xticks([])
        ax[3].set_yticks([])
        ax[3].set_ylabel("vad pred")

        self._fix_layout()

        if plot:
            plt.pause(0.1)

        return fig, ax

    def create_background_fig(self):
        """
        Init background
        plots melspec, waveform, and two vad_oh plots
        """
        return self.plot_vad_sample_animation(
            self.waveform, vad=self.vad.permute(1, 0), plot=False
        )

    def save_dynamic_video(self, vid_path, verbose=False):
        """# Dynamic figure"""

        # Init blank figure with only the relevant
        # axis
        fig = plt.Figure(figsize=self.figsize, dpi=self.dpi)
        fig, axs = plt.subplots(4, 1, figsize=self.figsize, dpi=self.dpi)
        for a in axs:
            a.axis("off")

        ax = axs[3]
        ax.set_xlim([0, self.vad.shape[0]])
        self._fix_layout()

        # Init dynamic stuff
        self.pred_activity_patches = self.init_bin_boxes(ax)
        self.pred_outline_patches = self.init_outline_patches(ax)
        self.current_step_lines = self.init_current_step_lines(ax)

        if verbose:
            pbar = tqdm(self.frames)
        else:
            pbar = self.frames
        # record animation with only the dynamic info
        moviewriter = animation.FFMpegWriter(
            fps=10  # , codec="libopenh264", extra_args=["-threads", "16"]
        )
        with moviewriter.saving(fig, vid_path, dpi=self.dpi):
            for t in pbar:
                _ = self.update(t)
                moviewriter.grab_frame()
        plt.close(fig)

    def init_bin_boxes(self, ax=None):
        """Vad-prediction boxes colored by speaker showing the activity of each bin"""
        colors = ["b", "orange"]
        bin_boxes = [[], []]
        for _ in self.bin_sizes:
            p1 = Rectangle(
                xy=(0, 0),
                width=0,
                height=0,
                color=colors[0],
                linewidth=2,
            )
            if ax is None:
                self.ax[3].add_patch(p1)
            else:
                ax.add_patch(p1)
            p2 = Rectangle(
                xy=(0, -1),
                width=0,
                height=0,
                color=colors[1],
                linewidth=2,
            )
            if ax is None:
                self.ax[3].add_patch(p2)
            else:
                ax.add_patch(p2)
            bin_boxes[0].append(p1)
            bin_boxes[1].append(p2)
        return bin_boxes

    def init_outline_patches(self, ax=None):
        """
        The outlines in vad-prediction window.

        Used to have a larger box around the entire window but given how
        we save the animation (overlay image on top of video) this is not
        qualitatively visible
        outer box
        bin boundary lines
        """
        patches = []
        for i, _ in enumerate(self.bin_sizes):
            linestyle = "dashed"
            if i == 3:  # last line is solid marking the end of window
                linestyle = "solid"
            if ax is None:
                (ln,) = self.ax[3].plot(
                    (),
                    (),
                    linestyle=linestyle,
                    linewidth=2,
                    color="k",
                )
            else:
                (ln,) = ax.plot(
                    (),
                    (),
                    linestyle=linestyle,
                    linewidth=2,
                    color="k",
                )
            patches.append(ln)
        return patches

    def init_current_step_lines(self, ax=None):
        """Show the current timestamp in the vad-pred window

        Omitted showin lines on spec/vad because they are not visible
        when saving the animation... However, they were visible in the
        live animation. But simpler to omit.
        """
        lines = []
        for i in [3]:
            if ax is None:
                (ln,) = self.ax[i].plot([], [], linewidth=2, color="r")
                ymin, ymax = self.ax[i].get_ylim()
                lines.append([ln, ymin, ymax])
            else:
                (ln,) = ax.plot([], [], linewidth=2, color="r")
                ymin, ymax = ax.get_ylim()
                lines.append([ln, ymin, ymax])
        return lines

    def update_bin_boxes(self, frame):
        """Update the bin-boxes see `self.init_bin_boxes`"""
        ret = []
        v = self.vad_label_oh[frame]
        for ch, ch_bins in enumerate(v):
            offset = 0
            for bin, val in enumerate(ch_bins):
                bin_width = self.bin_sizes[bin]
                x = frame + offset
                if val > 0:
                    self.pred_activity_patches[ch][bin].set_bounds(x, -ch, bin_width, 1)
                else:  # hide patch
                    self.pred_activity_patches[ch][bin].set_bounds(x, -ch, 0, 0)
                ret.append(self.pred_activity_patches[ch][bin])
                offset += bin_width
        return ret

    def update_outline_patches(self, frame):
        """Update the bin-boxes see `self.init_outline_patches`"""
        ret = []
        x = frame
        for i, bs in enumerate(self.bin_sizes):
            x += bs
            self.pred_outline_patches[i].set_data((x, x), (-1, 1))
            ret.append(self.pred_outline_patches[i])
        return ret

    def update_current_step_lines(self, frame):
        """Update the bin-boxes see `self.init_current_step_lines`"""
        ret = []
        for ll in self.current_step_lines:
            ll[0].set_data([frame, frame], [ll[1], ll[2]])
            ret.append(ll[0])
        return ret

    def update(self, frame):
        """updates the frame for the `FuncAnimation`"""
        ret = self.update_current_step_lines(frame)
        ret += self.update_bin_boxes(frame)
        ret += self.update_outline_patches(frame)
        return ret

    def ffmpeg_call(self, vid_path, wav_path, img_path, out_path):
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
            "-loop",
            "1",
            "-i",
            img_path,
            "-vcodec",
            "libopenh264",
            "-filter_complex",
            "overlay=0:shortest=1",
            out_path,
        ]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        p.communicate()

    def create_savepath_dir(self, savepath):
        Path(dirname(savepath)).mkdir(parents=True, exist_ok=True)

    def save_animation(self, savepath="test_overlay.mp4"):
        """Saves the animation to `savepath`"""

        # tmp files
        tmp_wav_path = "/tmp/test_audio.wav"
        tmp_vid_path = "/tmp/background_video.mp4"
        tmp_img_path = "/tmp/background.png"

        # Save waveform (temporary)
        torchaudio.save(
            tmp_wav_path, self.waveform.unsqueeze(0), sample_rate=self.sample_rate
        )

        # Save the static "background" image of the whole segment
        # We actually overlay the image on top of the video (wanted
        # the opposite but had trouble with transparency of the video frames)
        self.fig, self.ax = self.create_background_fig()
        self.fig.savefig(tmp_img_path, transparent=True, dpi=self.dpi)
        plt.close(self.fig)

        # Save the dynamic animation
        self.save_dynamic_video(tmp_vid_path)

        # Combine video, image and audio with ffmpeg
        self.create_savepath_dir(savepath)
        self.ffmpeg_call(tmp_vid_path, tmp_wav_path, tmp_img_path, savepath)

        # Remove temporary files
        remove(tmp_vid_path)
        remove(tmp_wav_path)
        remove(tmp_img_path)

    def animation(self, show=True):
        """live animation"""

        # Init figure (and static information)
        self.fig, self.ax = self.create_background_fig()

        # Init dynamic stuff
        self.pred_activity_patches = self.init_bin_boxes()
        self.pred_outline_patches = self.init_outline_patches()
        self.current_step_lines = self.init_current_step_lines()

        self.ani = FuncAnimation(
            self.fig,
            self.update,
            frames=self.frames,
            interval=100,
            blit=True,
        )
        if show:
            plt.show()
        return self.ani


# Not used kept for reference for the matplotlib jungle...
class AniAlts(VadPredAnimator):
    """
    These are steps I tried to speed up saving the animation.

    A 10 second clip took between 58 and 72 seconds for all methods
    which is due to the size of the static information...
    """

    def animate_imgs(self):
        from os import makedirs
        from os.path import join
        import shutil
        import subprocess

        cache = "/tmp/matplot_videos"
        makedirs(cache, exist_ok=True)

        # Use Agg backend for canvas
        for i, t in enumerate(tqdm(range(0, 1000, 10))):
            self.update(t)
            self.fig.savefig(join(cache, "file%02d.png" % i))

        subprocess.call(
            [
                "ffmpeg",
                "-framerate",
                "10",
                "-i",
                join(cache, "file%02d.png"),
                "-r",
                "30",
                "-pix_fmt",
                "yuv420p",
                "test_imgs.mp4",
            ]
        )
        shutil.rmtree(cache)

    def animate_cv2(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        import numpy as np
        import cv2

        frame_width, frame_height = self.fig.canvas.get_width_height()
        video = cv2.VideoWriter(
            "test_cv2.avi",
            # cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            cv2.VideoWriter_fourcc(*"H264"),
            # cv2.VideoWriter_fourcc(*"mp4v"),
            10,
            (frame_width, frame_height),
        )
        canvas = FigureCanvas(self.fig)
        for t in tqdm(range(0, 1000, 10)):
            self.update(t)
            canvas.draw()
            mat = np.array(canvas.renderer._renderer)
            mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)

            # write frame to video
            video.write(mat)
            # video.write(np.array(canvas.renderer._renderer))
        # close video writer
        cv2.destroyAllWindows()
        video.release()

    def animate_pipe(self):
        """
        # https://stackoverflow.com/questions/30965355/speedup-matplotlib-animation-to-video-file
        """
        import subprocess

        canvas_width, canvas_height = self.fig.canvas.get_width_height()

        # Open an ffmpeg process
        outf = "test_pipe.mp4"
        cmdstring = (
            "ffmpeg",
            "-y",
            "-r",
            "10",  # overwrite, 1fps
            "-s",
            "%dx%d" % (canvas_width, canvas_height),  # size of image string
            "-pix_fmt",
            "argb",  # format
            "-f",
            "rawvideo",
            "-i",
            "-",  # tell ffmpeg to expect raw video from the pipe
            "-vcodec",
            "mpeg4",
            "-threads",
            "16",
            outf,
        )  # output encoding
        p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

        for t in range(0, 1000, 10):
            self.update(t)
            self.fig.canvas.draw()
            string = self.fig.canvas.tostring_argb()
            # write to pipe
            p.stdin.write(string)

        # Finish up
        p.communicate()

    def animate_ffmpeg(self, savepath):
        """
        Takes around 70 seconds on 10 seconds audio...
        """
        # print("Makin a movie...")
        moviewriter = animation.FFMpegWriter(
            fps=10, codec="libopenh264", extra_args=["-threads", "16"]
        )
        with moviewriter.saving(self.fig, savepath, dpi=100):
            for t in tqdm(range(0, 1000, 10)):
                _ = self.update(t)
                moviewriter.grab_frame()

    def animate_imagick(self):
        from matplotlib.animation import ImageMagickWriter

        writer = ImageMagickWriter(
            fps=10
        )  # , codec="h264", bitrate=1500, extra_args=["-threads", "16"]
        ani = FuncAnimation(
            self.fig,
            self.update,
            frames=range(0, 1000, 10),
            interval=100,
            blit=True,
        )
        print("Saving ani...")
        ani.save("test_ani_imagick.mp4", writer=writer)
        # plt.show()
        # ani.save("test_ani.mp4")
        return ani


if __name__ == "__main__":
    from datasets_turntaking.dm_dialog_audio import quick_load_dataloader
    from datasets_turntaking.features.vad import VadProjection
    import time

    dloader = quick_load_dataloader()
    batch = next(iter(dloader))

    vad_labels = batch["vad_label"]
    print("vad_labels: ", tuple(vad_labels.shape), vad_labels.dtype)
    n_bins = 8
    emb = VadProjection(n_bins=n_bins)
    vad_label_oh = emb(vad_labels)
    print("vad_label_oh: ", tuple(vad_label_oh.shape), vad_label_oh.dtype)

    b = 1
    t = time.time()
    vp = VadPredAnimator(
        waveform=batch["waveform"][b],
        vad=batch["vad"][b],
        vad_label_oh=vad_label_oh[b].view(1000, 2, 4),
        bin_sizes=[20, 40, 60, 80],
    )
    # vp.animation()
    vp.save_animation("test.mp4")
    t = time.time() - t
    print(f"Animation took {round(t, 2)}s")

    # dpi 100: 5.62s
    # dpi 300: 12.25s

    # ani = vp.animate_ffmpeg_overlay()  #  ~7s
    # vp.animate_cv2()  # fastest ~58s
    # ani = vp.animate()
    # vp.animate_ffmpeg("test_ffmpeg.mp4")
    # vp.animate_funcblit()
    # ani = vp.animate_imagick()
    # vp.animate_pipe()
    # vp.animate_imgs()
