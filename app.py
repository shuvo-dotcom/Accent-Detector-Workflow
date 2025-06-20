import streamlit as st
from pathlib import Path
from pydub import AudioSegment
from speechbrain.pretrained import SpeakerRecognition

# Paths
BASE_DIR = Path(__file__).resolve().parent
REF_DIR = BASE_DIR / "ref_accents"
UPLOAD_DIR = BASE_DIR / "user_inputs"
UPLOAD_DIR.mkdir(exist_ok=True)

# Load model
st.title("🗣️ Accent Detection with SpeechBrain")
classifier = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=BASE_DIR / "pretrained_models/spkrec-ecapa-voxceleb"
)

# Upload
uploaded_file = st.file_uploader("Upload your `.wav` file", type=["wav"])
if uploaded_file:
    user_path = UPLOAD_DIR / uploaded_file.name
    with open(user_path, "wb") as f:
        f.write(uploaded_file.read())

    # Standardize uploaded audio
    st.audio(str(user_path), format="audio/wav")
    audio = AudioSegment.from_file(user_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(user_path, format="wav")

    # Compare
    st.info("🔍 Comparing with reference accents...")
    scores = {}
    for ref_file in sorted(REF_DIR.glob("*.wav")):
        accent = ref_file.stem
        try:
            score, _ = classifier.verify_files(
                str(ref_file.resolve()), str(user_path.resolve())
            )
            scores[accent] = float(score)
        except Exception as e:
            scores[accent] = 0.0
            st.warning(f"❌ Error comparing to {accent}: {e}")

    if scores:
        st.subheader("📊 Confidence Scores")
        st.write(scores)

        best_accent = max(scores, key=scores.get)
        confidence = scores[best_accent]
        st.success(f"🎯 **Predicted Accent:** `{best_accent.capitalize()}`")
        st.markdown(f"**Confidence Score:** `{confidence:.2f}`")
