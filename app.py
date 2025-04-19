import streamlit as st
import av
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import pandas as pd
import time
import cv2

# Set up Streamlit app
st.set_page_config(page_title="SafeVision AI", layout="centered")
st.title("🚨 SafeVision AI - Real-Time Safety Detection")
st.markdown("Live person detection using MediaPipe. Emotion detection temporarily disabled for cloud compatibility.")

# Sidebar
alert_enabled = st.sidebar.checkbox("Enable Alerts", value=True)
save_logs = st.sidebar.checkbox("Save Detection Logs")

# Initialize session state keys
if 'log' not in st.session_state:
    st.session_state['log'] = []

# Mediapipe face detection setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

class MediapipeProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.last_detected = "neutral"
        self.num_faces = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(rgb_img)

        self.num_faces = 0
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(img, (x, y), (x+w, y+h), (100, 200, 250), 2)
                cv2.putText(img, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 250), 2)
                self.num_faces += 1

                mock_emotion = "neutral"
                self.last_detected = mock_emotion

                if save_logs:
                    st.session_state['log'].append({
                        "time": time.strftime("%H:%M:%S"),
                        "emotion": mock_emotion,
                        "faces": self.num_faces
                    })

        return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="mediapipe-stream",
    video_processor_factory=MediapipeProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# Display results
if ctx.video_processor:
    emotion = ctx.video_processor.last_detected
    faces = ctx.video_processor.num_faces
    st.metric("🧍 Number of People Detected", faces)
    if emotion and alert_enabled and emotion in ["angry", "fear", "disgust"]:
        st.error(f"⚠️ Potential Threat Detected: {emotion.upper()}")

# Show logs
if save_logs and st.session_state['log']:
    st.markdown("## 📝 Detection Log")
    st.dataframe(pd.DataFrame(st.session_state['log']))

