"""Streamlit interface for classifying a test window or uploaded sensor CSV."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import HARNet

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING UPSTAIRS",
    "WALKING DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]
CHANNEL_NAMES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


@st.cache_resource
def load_model():
    """Load the selected trained checkpoint once per Streamlit process."""
    model = HARNet()
    model.load_state_dict(
        torch.load(
            ROOT / "results" / "best_har_model.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()
    return model


@st.cache_data
def prepared_test_data():
    """Load the test tensors and training-only normalization statistics."""
    npz = np.load(ROOT / "results" / "prepared.npz")
    from prepare import prepare

    tensors = prepare(str(ROOT / "data"), str(ROOT / "results"))
    return tensors


st.set_page_config(page_title="Human Activity Recognition", layout="wide")
st.title("Phone Motion Human Activity Recognition")

# Load the saved classifier and the same preprocessing values used in training.
model = load_model()
tensors = prepared_test_data()

# Let the user select a bundled window or provide a raw 128-by-9 CSV.
choice = st.radio(
    "Input source", ["Pick test window", "Upload 128 x 9 CSV"], horizontal=True
)
if choice == "Pick test window":
    index = st.number_input("Test window index", 0, tensors["X_test"].shape[0] - 1, 0)
    window = tensors["X_test"][int(index)].numpy()
else:
    uploaded = st.file_uploader("CSV with 128 rows and 9 sensor columns", type=["csv"])
    if uploaded is None:
        st.stop()
    frame = pd.read_csv(uploaded)
    if frame.shape != (128, 9):
        st.error(f"Expected shape (128, 9), received {frame.shape}.")
        st.stop()
    raw_window = frame.to_numpy(dtype=np.float32)
    window = (raw_window - tensors["mean"].numpy()) / tensors["std"].numpy()

# Inference is prediction-only: gradients are unnecessary in the app.
with torch.no_grad():
    logits, attention = model(torch.from_numpy(window.astype(np.float32)).unsqueeze(0))
    probabilities = torch.softmax(logits, dim=1)[0]

predicted = int(probabilities.argmax())
st.success(f"Predicted activity: **{ACTIVITY_NAMES[predicted]}**")

# Present sensor context and model output next to each other.
col1, col2 = st.columns(2)
with col1:
    st.subheader("Sensor signals")
    signal_df = pd.DataFrame(window, columns=CHANNEL_NAMES)
    signal_df.insert(0, "time_s", np.arange(128) / 50.0)
    st.line_chart(signal_df.set_index("time_s"))
with col2:
    st.subheader("Activity probabilities")
    st.bar_chart(
        pd.DataFrame(
            {
                "Probability": probabilities.numpy(),
            },
            index=ACTIVITY_NAMES,
        )
    )

st.subheader("Attention weights")
attention_df = pd.DataFrame(
    {
        "time_s": np.arange(128) / 50.0,
        "attention": attention[0].numpy(),
    }
).set_index("time_s")
st.area_chart(attention_df)
