# Accent-Detector-Workflow

Install Dependencies
virtualenv workenv
source workenv/bin/activate
pip install -r requirements.txt
brew install ffmpeg


Use Python 3.10

Run the App
streamlit run app.py

How It Works
1. Upload a .wav or .mp4 file.
2. The app extracts audio, compares it with stored reference accents using speechbrain.
3. The accent with the highest score is predicted.
4. Simultaneously, Whisper transcribes the audio and displays the transcript.

Hosted @
https://accent-detector-workflow.streamlit.app/

Examples
![Alt Text](https://github.com/shuvo-dotcom/Accent-Detector-Workflow/blob/main/Screenshot%202025-06-20%20at%209.45.11%E2%80%AFPM.png)
![Alt Text](https://github.com/shuvo-dotcom/Accent-Detector-Workflow/blob/main/Screenshot%202025-06-20%20at%209.45.33%E2%80%AFPM.png)

