import streamlit as st
import matplotlib as mpl
import soundfile

import torch
from pytorch_lightning import Trainer

from conv_ssl.model import VPModel
from conv_ssl.utils import to_device, everything_deterministic
from conv_ssl.plot_utils import plot_window

from datasets_turntaking import DialogAudioDM


mpl.use("agg")
st.set_page_config(layout="wide")

BATCH_SIZE = 10
NUM_WORKERS = 4
CHECKPOINT = "checkpoints/cpc/cpc_04_44.ckpt"

everything_deterministic()


def update_session_state():
    if "model" not in st.session_state:
        with st.spinner(text="Loading Model..."):
            st.session_state.model = load_model()
            st.success("Model loaded")

    if "dset" not in st.session_state or "dm" not in st.session_state:
        with st.spinner(text="Loading Data..."):
            st.session_state.dm = load_dataset(st.session_state.model)
            st.session_state.dset = st.session_state.dm.val_dset
            st.session_state.current_sample = st.session_state.dset[0]
            update_waveform()
            forward_pass()
            st.success("Data loaded")

    if "result" not in st.session_state:
        st.session_state.result = {}

    if "evaluation_result" not in st.session_state:
        st.session_state.evaluation_result = {}

    if "audio_file" not in st.session_state:
        st.session_state.audio_file = ""


def load_model():
    model = VPModel.load_from_checkpoint(CHECKPOINT)
    model = model.to("cuda")
    model.eval()
    return model


def load_dataset(model):
    data_conf = DialogAudioDM.load_config()
    dm = DialogAudioDM(
        datasets=data_conf["dataset"]["datasets"],
        type=data_conf["dataset"]["type"],
        audio_duration=data_conf["dataset"]["audio_duration"],
        audio_normalize=data_conf["dataset"]["audio_normalize"],
        audio_overlap=data_conf["dataset"]["audio_overlap"],
        sample_rate=data_conf["dataset"]["sample_rate"],
        vad_hz=model.frame_hz,
        vad_bin_times=model.conf["vad_projection"]["bin_times"],
        vad_history=data_conf["dataset"]["vad_history"],
        vad_history_times=data_conf["dataset"]["vad_history_times"],
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    dm.prepare_data()
    dm.setup()
    return dm


def update_waveform():
    filepath = "tmp.wav"
    sample_rate = 16000
    waveform = st.session_state.current_sample["waveform"][0].numpy()
    soundfile.write(filepath, waveform, sample_rate, "PCM_24")
    st.session_state.audio_file = filepath


def get_new_sample():
    idx = st.session_state.sample_idx
    st.session_state.current_sample = st.session_state.dset[idx]
    update_waveform()
    forward_pass()


def forward_pass():
    model = st.session_state.model
    batch = st.session_state.current_sample
    batch = to_device(batch, model.device)

    model.val_metric.reset()
    with torch.no_grad():
        _, out, batch = model.shared_step(batch)
        probs, pre_probs = model.get_next_speaker_probs(
            out["logits_vp"], vad=batch["vad"]
        )
        events = model.val_metric.extract_events(batch["vad"])
        model.val_metric.update(probs, None, events=events, bc_pre_probs=pre_probs)
        result = model.val_metric.compute()

    st.session_state.result = result

    b = 0
    fig, ax = plot_window(
        probs[b],
        vad=batch["vad"][b],
        hold=events["hold"][b],
        shift=events["shift"][b],
        pre_hold=events["pre_hold"][b],
        pre_shift=events["pre_shift"][b],
        backchannels=events["backchannels"][b],
        plot=False,
    )
    st.session_state.fig = fig
    st.session_state.ax = ax


def evaluate():
    trainer = Trainer(gpus=-1, limit_test_batches=50, deterministic=True)
    with st.spinner(text="Evaluation..."):
        result = trainer.test(
            st.session_state.model,
            dataloaders=st.session_state.dm.val_dataloader(),
            verbose=False,
        )

    st.session_state.evaluation_result = result[0]


if __name__ == "__main__":
    update_session_state()

    with st.container():
        st.markdown(
            """
            # Vad Projection

            Description blablba balblab ballablblabl alblablablalblabablablag blbalblblbalblabla

            28
            5000

            """
        )

    c1, c2 = st.columns([1, 3])
    with c1:
        st.button("Evaluation", on_click=evaluate)
    with c2:
        st.json(st.session_state.evaluation_result)

    # Window
    st.title("Window")
    current_sample = st.number_input(
        f"Sample IDX {len(st.session_state.dset)}",
        0,
        len(st.session_state.dset),
        value=0,
        step=1,
        key="sample_idx",
        on_change=get_new_sample,
    )

    if st.session_state.audio_file != "":
        st.audio(st.session_state.audio_file, format="audio/wav")

    st.pyplot(st.session_state.fig)

    if st.session_state.result != {}:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.write("Metrics")
            st.json(
                {
                    "F1": st.session_state.result["f1_weighted"],
                    "F1 pre": st.session_state.result["f1_pre_weighted"],
                    "BC": st.session_state.result["bc"],
                }
            )
        with c2:
            st.write("SHIFT")
            st.json(st.session_state.result["shift"])
            st.write("PRE SHIFT")
            st.json(st.session_state.result["pre_shift"])
        with c3:
            st.write("HOLD")
            st.json(st.session_state.result["hold"])
            st.write("PRE HOLD")
            st.json(st.session_state.result["pre_hold"])
