import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from utils import get_landmark_coords, calculate_normalized_distance, classify_emotion, save_emotions

eye_series, mouth_series = [], []
emotion_labels, timestamps = [], []
previous_eye, previous_mouth = None, None
SPIKE_THRESHOLD = 0.05

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    landmarks = results.multi_face_landmarks or []

    for face in landmarks:
        image_shape = frame.shape

        # Get required facial landmarks
        left_eye = get_landmark_coords(face.landmark, 159, image_shape)
        left_eye_lower = get_landmark_coords(face.landmark, 145, image_shape)
        mouth_upper = get_landmark_coords(face.landmark, 13, image_shape)
        mouth_lower = get_landmark_coords(face.landmark, 14, image_shape)
        eyebrow_left = get_landmark_coords(face.landmark, 295, image_shape)
        eye_left_top = get_landmark_coords(face.landmark, 159, image_shape)
        eyebrow_raise = calculate_normalized_distance(eyebrow_left, eye_left_top)
        chin = get_landmark_coords(face.landmark, 152, image_shape)
        jaw = get_landmark_coords(face.landmark, 17, image_shape)
        jaw_drop = calculate_normalized_distance(chin, jaw)
        nose_top = get_landmark_coords(face.landmark, 6, image_shape)
        nose_bottom = get_landmark_coords(face.landmark, 197, image_shape)
        nose_wrinkle = calculate_normalized_distance(nose_top, nose_bottom)

        # Calculate distances
        eye_openness = calculate_normalized_distance(left_eye, left_eye_lower)
        mouth_openness = calculate_normalized_distance(mouth_upper, mouth_lower)

        # Append series
        eye_series.append(eye_openness)
        mouth_series.append(mouth_openness)

        # Emotion classification
        landmarks_features = (eye_openness, mouth_openness, eyebrow_raise, jaw_drop, nose_wrinkle)
        emotion, score = classify_emotion(frame, landmarks_features)
        current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        emotion_labels.append(emotion)
        timestamps.append(current_timestamp)

        # Detect spikes
        spike_messages = []
        if eyebrow_raise > 0.05:
            emotion_labels.append("SURPRISE or ANGER")
        if jaw_drop > 0.07:
            emotion_labels.append("SHOCK")
        if nose_wrinkle < 0.01:
            emotion_labels.append("DISGUST")
        if previous_eye is not None and abs(eye_openness - previous_eye) > SPIKE_THRESHOLD:
            spike_messages.append(("SURPRISE spike in eye openness!", (255, 255, 0)))
            emotion_labels.append("SURPRISE")

        if previous_mouth is not None and abs(mouth_openness - previous_mouth) > SPIKE_THRESHOLD:
            spike_messages.append(("DISCOMFORT spike in mouth openness!", (0, 255, 255)))
            emotion_labels.append("DISCOMFORT")

        previous_eye = eye_openness
        previous_mouth = mouth_openness

        # Draw facial lines
        cv2.line(frame, left_eye, left_eye_lower, (255, 0, 0), 1)
        cv2.line(frame, mouth_upper, mouth_lower, (0, 255, 0), 1)

        # Dynamic overlay text
        y_pos = 30
        overlay_items = [
            (f"Eye: {eye_openness:.2f}", (255, 0, 0)),
            (f"Mouth: {mouth_openness:.2f}", (0, 255, 0)),
            (f"Emotion: {emotion} ({score:.2f})", (255, 255, 0)) if emotion else None
        ] + [(msg, color) for msg, color in spike_messages]

        for item in overlay_items:
            if item:
                text, color = item
                cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_pos += 20

        if emotion is not None and score is not None:
            cv2.putText(frame, f"Emotion: {emotion} ({score:.2f})", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Show frame
    cv2.imshow("LieLens - Live Feed", frame)

    # Save emotion data
    save_emotions(emotion_labels, timestamps)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
