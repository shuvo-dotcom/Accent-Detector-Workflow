import streamlit as st
import os
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import numpy as np
from moviepy import *
from speechbrain.pretrained import SpeakerRecognition
import whisper
import torch
import torchaudio
torchaudio.set_audio_backend("soundfile")
st.set_page_config(page_title="Accent Detector", layout="centered")
st.title("🧠 Accent Detector & Transcriber")
st.caption("Upload a WAV or MP4 file. We'll detect the accent and transcribe what was said.")

# Folders
audio_dir = Path("user_inputs")
ref_dir = Path("ref_accents")
audio_dir.mkdir(exist_ok=True)
ref_dir.mkdir(exist_ok=True)

# Upload
uploaded_file = st.file_uploader("🎙️ Upload WAV or MP4", type=["wav", "mp4"], help="Max 200MB")

if uploaded_file:
    # Save uploaded file
    user_path = audio_dir / uploaded_file.name
    with open(user_path, "wb") as f:
        f.write(uploaded_file.read())

    # Convert MP4 to WAV if needed
    if uploaded_file.name.endswith(".mp4"):
        st.info("🎞️ Extracting audio from video...")
        audio_path = user_path.with_suffix(".wav")
        clip = VideoFileClip(str(user_path))
        clip.audio.write_audiofile(str(audio_path), codec='pcm_s16le')
        clip.close()
        user_path.unlink()  # Remove original video
    else:
        audio_path = user_path

    st.audio(str(audio_path), format='audio/wav')

    st.info("🔍 Scanning for reference accents...")
    ref_files = list(ref_dir.glob("*.wav"))

    if not ref_files:
        st.warning("⚠️ No reference `.wav` files found in `ref_accents/` folder.")
    else:
        st.success(f"✅ Found {len(ref_files)} reference file(s).")

        with st.spinner("⏳ Please wait a few moments... We're analyzing your accent. 🚀"):
            # Accent Detection
            classifier = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec"
            )

            scores = {}
            for ref_file in ref_files:
                label = ref_file.stem.lower()
                try:
                    score, _ = classifier.verify_files(
                        str(ref_file),
                        str(audio_path)
                    )
                    scores[label] = score
                except Exception as e:
                    scores[label] = 0
                    st.error(f"❌ Error comparing to {label}: {e}")

            sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
            predicted = next(iter(sorted_scores))
            predicted_score = sorted_scores[predicted]
            if isinstance(predicted_score, torch.Tensor):
                predicted_score = predicted_score.item()

            # Display scores
            st.subheader("📊 Confidence Scores")
            printable_scores = {
                k: float(v.item()) if isinstance(v, torch.Tensor) else float(v)
                for k, v in sorted_scores.items()
            }
            st.json(printable_scores)

            st.success(f"🎯 **Predicted Accent:** `{predicted.capitalize()}`")
            st.markdown(f"**Confidence Score:** `{predicted_score:.2f}`")

            # Plot
            fig, ax = plt.subplots()
            ax.bar(printable_scores.keys(), printable_scores.values(), color='skyblue')
            ax.set_xlabel("Accent")
            ax.set_ylabel("Confidence Score")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

            # Transcription
            st.info("🧾 Transcribing audio with Whisper...")
            whisper_model = whisper.load_model("base")
            transcription_result = whisper_model.transcribe(str(audio_path))
            transcribed_text = transcription_result["text"]

            st.subheader("📝 Transcription")
            st.text_area("Here's what was said:", transcribed_text, height=200)

        # Cleanup
        audio_path.unlink()
