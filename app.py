import streamlit as st
import os
import tempfile
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from speechbrain.pretrained import SpeakerRecognition

# Set paths
REF_DIR = Path("ref_accents")
USER_DIR = Path("user_inputs")
USER_DIR.mkdir(exist_ok=True)
REF_DIR.mkdir(exist_ok=True)

# Load classifier once
classifier = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# App UI
st.title("🗣️ Accent Detector")
st.markdown("Upload a WAV file (max 200MB) and I’ll try to detect its accent based on reference voices.")
uploaded_file = st.file_uploader("Drag and drop file here", type=["wav"])

# Process user upload
if uploaded_file:
    user_path = USER_DIR / uploaded_file.name
    with open(user_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(str(user_path))

    # Scan reference accents
    st.info("🔍 Looking for reference accents...")
    found_refs = list(REF_DIR.glob("*.wav"))

    if not found_refs:
        st.error("No reference accents found in `ref_accents/` folder.")
    else:
        st.success(f"Found {len(found_refs)} reference file(s): {[f.name for f in found_refs]}")
        st.info("🧠 Comparing with reference accents...")

        scores = {}
        for ref_file in found_refs:
            label = ref_file.stem.lower()
            try:
                score, _ = classifier.verify_files(str(ref_file.resolve()), str(user_path.resolve()))
                scores[label] = score
            except Exception as e:
                scores[label] = 0
                st.error(f"❌ Error comparing to {label}: {e}")

        # Show score
        st.subheader("📊 Confidence Scores")
        st.json(scores)

        # Plot
        fig, ax = plt.subplots()
        ax.bar(scores.keys(), scores.values())
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        st.pyplot(fig)

        # Predict
        predicted = max(scores, key=scores.get)
        st.success(f"🎯 Predicted Accent: **{predicted.capitalize()}**")
        st.caption(f"Confidence Score: `{scores[predicted]:.2f}`")

# Cleanup optional: uncomment if you want auto-delete after run
shutil.rmtree(USER_DIR)
