import streamlit as st
import os
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import numpy as np
from speechbrain.inference import SpeakerRecognition  # updated from deprecated 'pretrained'
import torch

st.set_page_config(page_title="Accent Detector", layout="centered")
st.title("🧠 Accent Detector")
st.caption("Upload a WAV file and detect the accent by comparing with known references.")

# Paths
audio_dir = Path("user_inputs")
ref_dir = Path("ref_accents")

# Ensure folders exist
audio_dir.mkdir(exist_ok=True)
ref_dir.mkdir(exist_ok=True)

# File upload
uploaded_file = st.file_uploader("🎙️ Upload WAV File", type=["wav"], help="Limit 200MB per file • WAV format only")

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    user_path = audio_dir / uploaded_file.name
    with open(user_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info("🔍 Scanning for reference accents...")
    ref_files = list(ref_dir.glob("*.wav"))

    if not ref_files:
        st.warning("⚠️ No reference `.wav` files found in the `ref_accents/` directory.")
    else:
        st.success(f"✅ Found {len(ref_files)} reference file(s): {[f.name for f in ref_files]}")

        st.info("🧠 Comparing with reference accents...")

        # Load model
        classifier = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec"
        )

        # Compare and score
        scores = {}
        for ref_file in ref_files:
            label = ref_file.stem.lower()
            try:
                score, _ = classifier.verify_files(str(ref_file.resolve()), str(user_path.resolve()))
                scores[label] = score
            except Exception as e:
                scores[label] = 0
                st.error(f"❌ Error comparing to {label}: {e}")

        # Sort by score
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        predicted = next(iter(sorted_scores))
        predicted_score = sorted_scores[predicted]
        if isinstance(predicted_score, torch.Tensor):
            predicted_score = predicted_score.item()

        # Show confidence scores
        st.subheader("📊 Confidence Scores")
        printable_scores = {
            k: float(v.item()) if isinstance(v, torch.Tensor) else float(v)
            for k, v in sorted_scores.items()
        }
        st.json(printable_scores)

        # Highlight result
        st.success(f"🎯 **Predicted Accent:** `{predicted.capitalize()}`")
        st.markdown(f"**Confidence Score:** `{predicted_score:.2f}`")

        # Plot results
        labels = list(printable_scores.keys())
        values = list(printable_scores.values())

        fig, ax = plt.subplots()
        ax.bar(labels, values, color='skyblue')
        ax.set_xlabel("Accent")
        ax.set_ylabel("Confidence Score")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        # Optional: remove uploaded file
        user_path.unlink()
