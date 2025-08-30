import numpy as np
from fer import FER
import csv
import cv2
from deepface import DeepFace

emotion_detector = FER()

def classify_emotion(frame, landmarks_features=None):
    try:
        resized_frame = cv2.resize(frame, (300, 300))
        result = DeepFace.analyze(resized_frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        emotions_dict = result[0]['emotion']
        score = emotions_dict.get(emotion.lower(), 0) / 100

        # ---- Hybrid logic ----
        if landmarks_features:
            eye_openness, mouth_openness, eyebrow_raise, jaw_drop, nose_wrinkle = landmarks_features

            if emotion == "NEUTRAL":
                # Override neutral if landmarks suggest fear
                if eyebrow_raise > 0.06 and eye_openness > 0.045 and mouth_openness < 0.05:
                    return "FEAR", 0.9
                if jaw_drop > 0.07:
                    return "SHOCK", 0.9

        return emotion.upper(), score

    except Exception as e:
        print("Emotion classification failed:", e)
        return None, None

def save_emotions(emotion_labels, timestamps):
    with open('emotion_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Timestamp", "Emotion"])
        for timestamp, emotion in zip(timestamps, emotion_labels):
            writer.writerow([timestamp, emotion])

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def get_landmark_coords(landmarks, idx, image_shape):
    h, w = image_shape[:2]
    point = landmarks[idx]
    return int(point.x * w), int(point.y * h)


def calculate_normalized_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))
