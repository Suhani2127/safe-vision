import streamlit as st
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from fer import FER
import pandas as pd
import time

# Set up Streamlit app
st.set_page_config(page_title="SafeVision AI", layout="centered")
st.title("🚨 SafeVision AI - Real-Time Safety Detection")
st.markdown("Live emotion detection using your webcam feed. All processing is done in-browser using FER (Facial Expression Recognition).")

# Sidebar
alert_enabled = st.sidebar.checkbox("Enable Alerts", value=True)
save_logs = st.sidebar.checkbox("Save Detection Logs")

# Init logs
if "log" not in st.session_state:
    st.session_state.log = []

# FER detector
emotion_detector = FER(mtcnn=True)

class EmotionProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.last_detected = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        result = emotion_detector.detect_emotions(img)

        if result:
            box = result[0]['box']
            emotions = result[0]['emotions']
            dominant_emotion = max(emotions, key=emotions.get)
            color = (0, 255, 0) if dominant_emotion in ["happy", "neutral"] else (0, 0, 255)

            cv2 = __import__('cv2')  # safely import only inside class to avoid streamlit cloud crash
            cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
            cv2.putText(img, dominant_emotion, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            self.last_detected = dominant_emotion

            if save_logs:
                st.session_state.log.append({
                    "time": time.strftime("%H:%M:%S"),
                    "emotion": dominant_emotion
                })

        return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="emotion-stream",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

if ctx.video_processor:
    emotion = ctx.video_processor.last_detected
    if emotion and alert_enabled and emotion in ["angry", "fear", "disgust"]:
        st.error(f"⚠️ Potential Threat Detected: {emotion.upper()}")

# Show logs
if save_logs and st.session_state.log:
    st.markdown("## 📝 Detection Log")
    st.dataframe(pd.DataFrame(st.session_state.log))

