import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import face_recognition
import numpy as np
import random
import time

# Page config
st.set_page_config(page_title="SafeVision AI", layout="centered")
st.title("🚨 SafeVision AI")
st.markdown("Live face detection + emergency alert system")

# Session state setup
if "emergency_contacts" not in st.session_state:
    st.session_state["emergency_contacts"] = ["911", "mom@example.com"]

if "log" not in st.session_state:
    st.session_state["log"] = []

if "alert_triggered" not in st.session_state:
    st.session_state["alert_triggered"] = False

# UI
st.sidebar.header("Settings")
alert_enabled = st.sidebar.checkbox("Enable Emergency Alerts", value=True)
mock_emotion = st.sidebar.selectbox("Your Current Emotion", ["neutral", "happy", "sad", "fear", "angry"])

# Alert display function
def trigger_emergency_alert(faces_detected, emotion):
    if not st.session_state["alert_triggered"]:
        st.session_state["alert_triggered"] = True
        st.error(f"🚨 Emergency Triggered! {faces_detected} people + emotion: {emotion}")
        for contact in st.session_state["emergency_contacts"]:
            st.write(f"📨 Alert sent to: {contact}")
        st.session_state["log"].append({
            "time": time.strftime("%H:%M:%S"),
            "people": faces_detected,
            "emotion": emotion
        })

# Video processor
class FaceDetector(VideoProcessorBase):
    def __init__(self):
        self.faces_detected = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_img = img[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_img)
        self.faces_detected = len(face_locations)

        for (top, right, bottom, left) in face_locations:
            color = (0, 255, 0)
            img = cv2.rectangle(img, (left, top), (right, bottom), color, 2)

        if alert_enabled and self.faces_detected >= 3 and mock_emotion in ["sad", "fear"]:
            trigger_emergency_alert(self.faces_detected, mock_emotion)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit webcam streamer
ctx = webrtc_streamer(
    key="face-detect-stream",
    video_processor_factory=FaceDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Metrics
if ctx.video_processor:
    st.metric("🧍 Faces Detected", ctx.video_processor.faces_detected)
    st.metric("🧠 Your Emotion", mock_emotion.capitalize())

# Log display
if st.session_state["log"]:
    st.markdown("## 🔐 Emergency Alert Log")
    for entry in st.session_state["log"]:
        st.write(entry)

