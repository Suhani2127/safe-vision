import streamlit as st
import cv2
import numpy as np
from fer import FER
import time

# Set up Streamlit page
st.set_page_config(page_title="SafeVision AI", layout="centered")
st.title("🚨 SafeVision AI - Real-Time Safety Detection")
st.markdown("Monitors webcam feed for nearby people and emotions to enhance personal safety.")

# Sidebar settings
alert_enabled = st.sidebar.checkbox("Enable Alerts", value=True)
save_logs = st.sidebar.checkbox("Save Detection Logs")

# Initialize FER detector
detector = FER(mtcnn=True)

# Set up session state for logs
if "log" not in st.session_state:
    st.session_state.log = []

# Capture video feed
cap = cv2.VideoCapture(0)
stframe = st.empty()
st.warning("Camera will turn on. Click 'Stop' in top-right when done.")

stop_time = time.time() + 60  # run for 60 seconds

while time.time() < stop_time:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access webcam")
        break

    # Resize for faster detection
    resized_frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Detect emotion using FER
    emotion_results = detector.detect_emotions(rgb_frame)
    dominant_emotion = "Unknown"

    if emotion_results:
        top_result = emotion_results[0]
        emotions = top_result["emotions"]
        dominant_emotion = max(emotions, key=emotions.get)
        (x, y, w, h) = top_result["box"]

        # Draw rectangle and label
        cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(resized_frame, f"{dominant_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0) if dominant_emotion in ["happy", "neutral"] else (0, 0, 255), 2)

    # Streamlit alert if threat emotion
    if alert_enabled and dominant_emotion in ["angry", "fear", "disgust"]:
        st.error(f"⚠️ Potential Threat Detected: {dominant_emotion.upper()}")

    # Save detection log
    if save_logs:
        st.session_state.log.append({
            "time": time.strftime("%H:%M:%S"),
            "emotion": dominant_emotion
        })

    # Display camera feed
    stframe.image(resized_frame, channels="BGR")

cap.release()

# Show log if selected
if save_logs and st.session_state.log:
    st.markdown("## 📝 Detection Log")
    st.dataframe(st.session_state.log)
