import streamlit as st
from pathlib import Path
from pydub import AudioSegment
from speechbrain.pretrained import SpeakerRecognition

# Set up directory paths
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

# Upload user audio
uploaded_file = st.file_uploader("Upload your `.wav` file", type=["wav"])
if uploaded_file:
    user_path = UPLOAD_DIR / uploaded_file.name
    with open(user_path, "wb") as f:
        f.write(uploaded_file.read())

    # Normalize audio to 16kHz mono
    audio = AudioSegment.from_file(user_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(user_path, format="wav")

    st.audio(str(user_path), format="audio/wav")

    # Debug: list reference files
    st.info("🔍 Looking for reference accents...")
    found_refs = list(REF_DIR.glob("*.wav"))
    if not found_refs:
        st.error("No reference `.wav` files found in `ref_accents/` folder.")
    else:
        st.success(f"Found {len(found_refs)} reference file(s): {[f.name for f in found_refs]}")

        # Compare with references
        scores = {}
        for ref_file in found_refs:
            accent = ref_file.stem
            try:
                score, _ = classifier.verify_files(
                    str(ref_file.resolve()),
                    str(user_path.resolve())
                )
                scores[accent] = float(score)
            except Exception as e:
                scores[accent] = 0.0
                st.error(f"❌ Error comparing to {accent}: {e}")

        # Output result
        st.subheader("📊 Confidence Scores")
        st.json(scores)

        best_accent = max(scores, key=scores.get)
        confidence = scores[best_accent]
        st.success(f"🎯 **Predicted Accent:** `{best_accent.capitalize()}`")
        st.markdown(f"**Confidence Score:** `{confidence:.2f}`")
