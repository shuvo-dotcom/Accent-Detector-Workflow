import streamlit as st
import tempfile
import os
from moviepy import *
from speechbrain.pretrained import SpeakerRecognition
import matplotlib.pyplot as plt
st.set_page_config(page_title="🎙️ English Accent Classifier", layout="centered")
st.title("🗣️ English Accent Detection (Open-Source)")
st.markdown("Upload an `.mp4` video and detect if the speaker has a **British**, **American**, or **Australian** English accent.")
uploaded_file = st.file_uploader("🎬 Upload your video file (max 20MB)", type=["mp4"])
MAX_SIZE_MB = 20
if uploaded_file:
    uploaded_file.seek(0, os.SEEK_END)
    file_size_mb = uploaded_file.tell() / (1024 * 1024)
    uploaded_file.seek(0)

    if file_size_mb > MAX_SIZE_MB:
        st.error("⚠️ File is too large! Please upload a file smaller than 20MB.")
        st.stop()

if uploaded_file:
    st.success("✅ File uploaded successfully.")
    if st.button("🔍 Process Video"):
        with st.spinner("⏳ Extracting audio and analyzing accent..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
                tmp_vid.write(uploaded_file.read())
                video_path = tmp_vid.name
            audio_path = video_path.replace(".mp4", ".wav")
            try:
                clip = VideoFileClip(video_path)
                clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
            except Exception as e:
                st.error(f"Failed to extract audio: {e}")
                st.stop()
            classifier = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="tmp/speechbrain"
            )
            reference_accents = {
                "British": "ref_accents/british.wav",
                "American": "ref_accents/american.wav",
                "Australian": "ref_accents/australian.wav"
            }
            scores = {}
            for accent, ref_path in reference_accents.items():
                try:
                    score = classifier.verify_files(audio_path, ref_path)[0].item()
                    scores[accent] = score
                except Exception as e:
                    scores[accent] = 0.0
            predicted = max(scores.items(), key=lambda x: x[1])

        st.success("✅ Accent detected!")
        st.markdown(f"### 🎯 **Predicted Accent:** `{predicted[0]}`")
        st.markdown(f"**Confidence Score:** `{predicted[1]:.2f}`")
        st.markdown("### 📊 Confidence Scores")

        fig, ax = plt.subplots()
        ax.bar(scores.keys(), scores.values(), color="skyblue")
        ax.set_ylabel("Score")
        ax.set_ylim([0, 1])
        st.pyplot(fig)

        os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
