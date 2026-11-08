from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


MODEL_PATH = Path(__file__).resolve().with_name("hand_landmarker.task")


def create_hand_landmarker():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"MediaPipe hand model not found: {MODEL_PATH}. "
            "Download hand_landmarker.task from "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hand_landmarks(hand_landmarker, frame, timestamp_ms):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return hand_landmarker.detect_for_video(image, timestamp_ms)


def landmarks_to_csv_row(label, landmarks):
    row = [label]
    for landmark in landmarks:
        row.extend([landmark.x, landmark.y, landmark.z])
    return row


def draw_hand_landmarks(frame, results):
    height, width = frame.shape[:2]

    for landmarks in results.hand_landmarks:
        points = [
            (
                max(0, min(width - 1, round(landmark.x * width))),
                max(0, min(height - 1, round(landmark.y * height))),
            )
            for landmark in landmarks
        ]

        for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(
                frame,
                points[connection.start],
                points[connection.end],
                (0, 255, 0),
                2,
            )

        for point in points:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)
