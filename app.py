import streamlit as st
import os
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from speechbrain.pretrained import SpeakerRecognition

st.set_page_config(page_title="Accent Detector", layout="centered")
st.title("🧠 Accent Detector")
st.caption("Upload a WAV file and detect the accent by comparing with known references.")

# File upload
uploaded_file = st.file_uploader("Drag and drop file here", type=["wav"], help="Limit 200MB per file • WAV")
audio_dir = Path("user_inputs")
ref_dir = Path("ref_accents")

# Ensure folders exist
audio_dir.mkdir(exist_ok=True)
ref_dir.mkdir(exist_ok=True)

# Save uploaded file
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    user_path = audio_dir / uploaded_file.name
    with open(user_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info("🔍 Looking for reference accents...")
    ref_files = list(ref_dir.glob("*.wav"))

    if len(ref_files) == 0:
        st.warning("⚠️ No reference accents found in `ref_accents/` folder.")
    else:
        st.success(f"Found {len(ref_files)} reference file(s): {[f.name for f in ref_files]}")

        st.info("🧠 Comparing with reference accents...")

        # Load SpeechBrain speaker recognition model
        classifier = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec"
        )

        scores = {}

        for ref_file in ref_files:
            label = ref_file.stem.lower()
            try:
                # Use real filesystem paths
                score, _ = classifier.verify_files(
                    str(ref_file.resolve()),
                    str(user_path.resolve())
                )
                scores[label] = score
            except Exception as e:
                scores[label] = 0
                st.error(f"❌ Error comparing to {label}: {e}")

        # Sort and display results
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        predicted = next(iter(sorted_scores))

        st.subheader("📊 Confidence Scores")
        st.json(sorted_scores)

        st.success(f"🎯 **Predicted Accent:** `{predicted.capitalize()}`")
        st.markdown(f"**Confidence Score:** `{sorted_scores[predicted]:.2f}`")

        # Plot
        fig, ax = plt.subplots()
        ax.bar(sorted_scores.keys(), sorted_scores.values())
        ax.set_xlabel("Accent")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        # Optional: Clean up uploaded file
        user_path.unlink()
