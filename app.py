import streamlit as st
import os
from pathlib import Path
import torchaudio
from speechbrain.pretrained import SpeakerRecognition

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
REF_DIR = BASE_DIR / "ref_accents"
UPLOAD_DIR = BASE_DIR / "user_inputs"
UPLOAD_DIR.mkdir(exist_ok=True)

# Load the SpeechBrain model
st.title("🗣️ Accent Detection with SpeechBrain")

st.markdown("Upload a `.wav` file. It will be compared against reference accents.")
classifier = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=BASE_DIR / "pretrained_models/spkrec-ecapa-voxceleb"
)

# Upload audio
uploaded_file = st.file_uploader("Upload your voice sample (.wav only)", type=["wav"])

if uploaded_file:
    user_path = UPLOAD_DIR / uploaded_file.name
    with open(user_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(str(user_path), format="audio/wav")

    # Verify against each reference file
    scores = {}
    for ref_file in sorted(REF_DIR.glob("*.wav")):
        accent = ref_file.stem
        try:
            score, _ = classifier.verify_files(str(ref_file), str(user_path))
            scores[accent] = float(score)
        except Exception as e:
            scores[accent] = 0.0
            st.warning(f"Error comparing to {accent}: {e}")

    if scores:
        st.subheader("📊 Confidence Scores")
        st.write(scores)

        best_accent = max(scores, key=scores.get)
        confidence = scores[best_accent]
        st.success(f"🎯 **Predicted Accent:** `{best_accent.capitalize()}`")
        st.markdown(f"**Confidence Score:** `{confidence:.2f}`")
