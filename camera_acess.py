import cv2
import os
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model file path for MediaPipe 1.0+ HandLandmarker
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Ensure the model file is present
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model file...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model file downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model file: {e}")
        exit(1)

# Initialize HandLandmarker Options
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

# Create detector
detector = vision.HandLandmarker.create_from_options(options)

# Hand skeleton connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm base
]

# Open webcam feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Webcam running at: {width}x{height} @ {fps} FPS")

cv2.namedWindow('Left and Right Palm Recognition', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture video frame.")
        break

    # Mirror flip frame horizontally
    flipped_frame = cv2.flip(frame, 1)

    # Convert BGR frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect hand landmarks
    detection_result = detector.detect(mp_image)

    h, w, _ = flipped_frame.shape

    # Process each detected hand
    if detection_result.hand_landmarks and detection_result.handedness:
        for landmarks, handedness in zip(detection_result.hand_landmarks, detection_result.handedness):
            # MediaPipe category_name for mirrored image:
            # "Left" in mirrored view -> Physical Right hand on screen
            # "Right" in mirrored view -> Physical Left hand on screen
            category = handedness[0]
            detected_label = category.category_name
            score = category.score

            hand_label = "Right Hand" if detected_label == "Left" else "Left Hand"

            # Color coding: Left Hand = Cyan, Right Hand = Bright Green
            if hand_label == "Left Hand":
                hand_color = (255, 255, 0)   # Cyan BGR
                text_color = (255, 255, 0)
            else:
                hand_color = (0, 255, 0)     # Bright Green BGR
                text_color = (0, 255, 0)

            # Convert normalized landmarks to pixel coordinates
            px_coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            x_coords = [pt[0] for pt in px_coords]
            y_coords = [pt[1] for pt in px_coords]

            # Calculate Bounding Box
            xmin, xmax = max(min(x_coords) - 15, 0), min(max(x_coords) + 15, w)
            ymin, ymax = max(min(y_coords) - 15, 0), min(max(y_coords) + 15, h)

            # Draw Hand Bounding Box
            cv2.rectangle(flipped_frame, (xmin, ymin), (xmax, ymax), hand_color, 2)

            # Compute Palm center (Wrist: 0, MCP joints: 1, 5, 9, 13, 17)
            palm_indices = [0, 1, 5, 9, 13, 17]
            palm_x = int(sum([px_coords[i][0] for i in palm_indices]) / len(palm_indices))
            palm_y = int(sum([px_coords[i][1] for i in palm_indices]) / len(palm_indices))

            # Draw Palm center circle & label
            cv2.circle(flipped_frame, (palm_x, palm_y), 12, hand_color, -1)
            cv2.circle(flipped_frame, (palm_x, palm_y), 16, (255, 255, 255), 2)
            cv2.putText(flipped_frame, "PALM", (palm_x - 20, palm_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw Skeleton Connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                pt1 = px_coords[start_idx]
                pt2 = px_coords[end_idx]
                cv2.line(flipped_frame, pt1, pt2, (220, 220, 220), 2)

            # Draw Skeleton Landmark Joints
            for pt in px_coords:
                cv2.circle(flipped_frame, pt, 4, hand_color, -1)

            # Render text banner for Hand Label and Confidence Score
            label_text = f"{hand_label} ({int(score * 100)}%)"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            banner_ymin = max(ymin - 35, 0)
            cv2.rectangle(flipped_frame, (xmin, banner_ymin), (xmin + text_w + 10, banner_ymin + text_h + 10), (20, 20, 20), -1)
            cv2.putText(flipped_frame, label_text, (xmin + 5, banner_ymin + text_h + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

    # Resize image to fit window
    _, _, win_w, win_h = cv2.getWindowImageRect('Left and Right Palm Recognition')
    if win_w > 0 and win_h > 0:
        display_frame = cv2.resize(flipped_frame, (win_w, win_h))
    else:
        display_frame = flipped_frame

    cv2.imshow('Left and Right Palm Recognition', display_frame)

    # Press 'x' or 'q' to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('x') or key == ord('q'):
        break

detector.close()
cap.release()
cv2.destroyAllWindows()