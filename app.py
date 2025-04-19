import cv2
import mediapipe as mp
from deepface import DeepFace  # For emotion detection
import numpy as np

# Initialize MediaPipe for face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Set up webcam feed
cap = cv2.VideoCapture(0)

def detect_emotion(face_image):
    """Detect emotion from a face image."""
    try:
        # DeepFace analysis for emotion
        analysis = DeepFace.analyze(face_image, actions=['emotion'])
        return analysis[0]['dominant_emotion']
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return "Neutral"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (MediaPipe uses RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract face for emotion detection
            face_image = frame[y:y + h, x:x + w]
            emotion = detect_emotion(face_image)

            # Display emotion
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Trigger alert for aggressive emotion
            if emotion in ['angry', 'fear']:
                cv2.putText(frame, "ALERT: Aggressive Emotion", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('SafeVision AI', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


