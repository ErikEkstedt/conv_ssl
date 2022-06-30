from argparse import ArgumentParser
import streamlit as st
import torch
import torchaudio
import json
from scipy.io.wavfile import read

import textgrids

from conv_ssl.model import VPModel
from conv_ssl.utils import (
    everything_deterministic,
    get_tg_vad_list,
    load_waveform,
    read_json,
    read_txt,
)
from conv_ssl.evaluation.duration import read_text_grid
from conv_ssl.evaluation.evaluation_phrases import plot_sample

everything_deterministic()


CHECKPOINT = "example/cpc_48_50hz_15gqq5s5.ckpt"
SAMPLE_RATE = 16000
TG_TMP_PATH = "tmp_textgrid.TextGrid"


EX_WAV = "example/student_long_female_en-US-Wavenet-G.wav"
EX_TG = "example/student_long_female_en-US-Wavenet-G.TextGrid"
EX_VA = "example/vad_list.json"


@st.cache
def load_model(checkpoint=CHECKPOINT):
    model = VPModel.load_from_checkpoint(checkpoint)
    model = model.eval()
    if torch.cuda.is_available():
        _ = model.to("cuda")
    return model


def load_vad_list(vad_list_data):
    vad_list = json.loads(vad_list_data.getvalue().decode("utf-8"))
    st.session_state.vad_list = vad_list


def load_textgrid(tg_data):
    tg_read = tg_data.getvalue().decode("utf-8")
    with open(TG_TMP_PATH, "w", encoding="utf-8") as f:
        f.write(tg_read)
    tg = read_text_grid(TG_TMP_PATH)
    vad_list = get_tg_vad_list(tg)
    st.session_state.tg = tg
    st.session_state.vad_list = vad_list


def run_model():
    if "waveform" in st.session_state and "vad_list" in st.session_state:
        sample = st.session_state.model.load_sample(
            st.session_state.waveform, st.session_state.vad_list
        )

        if "tg" in st.session_state and st.session_state.tg is not None:
            if "words" in st.session_state.tg:
                sample["words"] = st.session_state.tg["words"]

            if "phones" in st.session_state.tg:
                sample["phones"] = st.session_state.tg["phones"]

        loss, out, probs, sample = st.session_state.model.output(sample)
        # Save
        data = {
            "loss": {"vp": loss["vp"].item(), "frames": loss["frames"].tolist()},
            "probs": out["logits_vp"].softmax(-1).tolist(),
            "labels": out["va_labels"].tolist(),
            "p": probs["p"].tolist(),
            "p_bc": probs["bc_prediction"].tolist(),
        }
        st.session_state.output_data = data

        fig, ax = plot_sample(
            probs["p"][0, :, 0],
            sample,
            sample_rate=st.session_state.model.sample_rate,
            frame_hz=st.session_state.model.frame_hz,
        )
        st.session_state.fig = fig


def sample():
    waveform, _ = load_waveform(
        EX_WAV, sample_rate=SAMPLE_RATE, normalize=True, mono=True
    )
    tg = read_text_grid(EX_TG)
    vad_list = get_tg_vad_list(tg)
    sample = st.session_state.model.load_sample(waveform, vad_list)
    sample["words"] = tg["words"]
    sample["phones"] = tg["phones"]

    loss, out, probs, sample = st.session_state.model.output(sample)
    # Save
    data = {
        "loss": {"vp": loss["vp"].item(), "frames": loss["frames"].tolist()},
        "probs": out["logits_vp"].softmax(-1).tolist(),
        "labels": out["va_labels"].tolist(),
        "p": probs["p"].tolist(),
        "p_bc": probs["bc_prediction"].tolist(),
    }
    st.session_state.output_data = data
    fig, ax = plot_sample(
        probs["p"][0, :, 0],
        sample,
        sample_rate=st.session_state.model.sample_rate,
        frame_hz=st.session_state.model.frame_hz,
    )
    st.session_state.fig = fig


def clear():
    st.session_state.fig = None


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


if __name__ == "__main__":

    if check_password():
        with st.sidebar:
            st.header("Sample")

            with open(EX_WAV, "rb") as f:
                st.download_button(
                    "Download Wav", f, file_name="sample.wav", mime="audio/wav"
                )

            # st.subheader("VA List")
            # st.text(read_json(EX_VA), expanded=False)
            with open(EX_TG, "rb") as f:
                st.download_button(
                    "Download TextGrid", f, file_name="sample_tg.TextGrid"
                )

            with open(EX_VA, "rb") as f:
                st.download_button(
                    "Download VA list",
                    f,
                    file_name="sample_va.json",
                    mime="application/json",
                )

            st.subheader("Inspect")
            with st.expander("TextGrid"):
                with open(EX_TG, "r", encoding="utf-8") as f:
                    st.text(f.read())
            with st.expander("VA List"):
                st.json(read_json(EX_VA), expanded=True)

        if "model" not in st.session_state:
            st.session_state.model = load_model()

        if "output_data" not in st.session_state:
            st.session_state.output_data = None

        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                audio = st.file_uploader("Audio", type="wav")
                if audio is not None:
                    st.session_state.waveform, _ = load_waveform(
                        audio, sample_rate=SAMPLE_RATE, normalize=True, mono=True
                    )

            with col2:
                tg_data = st.file_uploader("TextGrid", type="TextGrid")
                if tg_data is not None:
                    load_textgrid(tg_data)

            with col3:
                vad_list_data = st.file_uploader("VA List", type="json")
                if vad_list_data is not None:
                    load_vad_list(vad_list_data)

        st.audio(audio)
        with st.container():
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.button(label="Run model", on_click=run_model)
            with c2:
                st.button(label="Run Sample", on_click=sample)
            with c3:
                st.button(label="Clear", on_click=clear)
            with c4:
                st.download_button(
                    label="Download",
                    data=json.dumps(st.session_state.output_data),
                    file_name="vap_output.json",
                    mime="application/json",
                )

        if "fig" in st.session_state and st.session_state.fig is not None:
            st.pyplot(st.session_state.fig)
