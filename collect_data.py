import csv
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from hand_tracking import (
    create_hand_landmarker,
    detect_hand_landmarks,
    draw_hand_landmarks,
)
from sequence_dataset import (
    NUM_FRAMES,
    POSE_FEATURES_PER_FRAME,
    SEQUENCE_CSV,
    add_velocity_features,
    landmarks_to_features,
)
from thumb_gesture import (
    MIN_CAPTURES_PER_CLASS,
    OTHER_GESTURE_LABEL,
    THUMB_DATASET_PATH,
    THUMB_MODEL_PATH,
    THUMB_UP_LABEL,
    ThumbGestureClassifier,
    append_capture,
    train_thumb_classifier,
)


CSV_PATH = Path(__file__).resolve().with_name(SEQUENCE_CSV)
SAMPLE_RATE = 15
COUNTDOWN_SECONDS = 0
CAPTURE_TIMEOUT_SECONDS = 8
THUMB_CAPTURE_SECONDS = 1


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open webcam.")

    last_timestamp_ms = 0
    session_id = uuid.uuid4().hex
    selected_label = None
    thumb_capture_label = None
    countdown_end = None
    capture_deadline = None
    thumb_capture_deadline = None
    next_sample_at = 0.0
    frames = []
    thumb_frames = []

    try:
        with create_hand_landmarker() as hand_landmarker:
            print(f"Recording gesture sequences to {CSV_PATH.name}.")
            print(
                "Press A-Z to record one example. Hold static letters; "
                "draw the full motion for moving letters."
            )
            print(
                "Record every letter in each session. Run this script again "
                "for a separate validation session."
            )
            print(
                "Train thumb detection: hold thumbs-up and press Space to "
                "capture; show another hand pose and press 0 to capture a "
                "negative example. Capture at least "
                f"{MIN_CAPTURES_PER_CLASS} separate examples of each pose."
            )
            thumb_classifier = ThumbGestureClassifier()
            print(
                "A trained thumb classifier is loaded."
                if thumb_classifier.model is not None
                else f"No trained thumb model yet. It will be saved as "
                f"{THUMB_MODEL_PATH.name}."
            )
            print("Press '/' to quit.")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                now = time.monotonic()
                timestamp_ms = max(
                    time.monotonic_ns() // 1_000_000,
                    last_timestamp_ms + 1,
                )
                last_timestamp_ms = timestamp_ms
                results = detect_hand_landmarks(hand_landmarker, frame, timestamp_ms)
                draw_hand_landmarks(frame, results)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("/"):
                    break

                if (
                    countdown_end is None
                    and capture_deadline is None
                    and thumb_capture_deadline is None
                ):
                    if key == ord(" "):
                        thumb_capture_label = THUMB_UP_LABEL
                        thumb_frames = []
                        thumb_capture_deadline = now + THUMB_CAPTURE_SECONDS
                        next_sample_at = now
                        print("Capturing thumbs-up example...")
                    elif key == ord("0"):
                        thumb_capture_label = OTHER_GESTURE_LABEL
                        thumb_frames = []
                        thumb_capture_deadline = now + THUMB_CAPTURE_SECONDS
                        next_sample_at = now
                        print("Capturing other-pose example...")

                if (
                    countdown_end is None
                    and capture_deadline is None
                    and thumb_capture_deadline is None
                ):
                    letter_key = key | 0x20
                    if ord("a") <= letter_key <= ord("z"):
                        selected_label = letter_key - ord("a")
                        countdown_end = now + COUNTDOWN_SECONDS
                        print(
                            f"Preparing to record "
                            f"{chr(selected_label + ord('A'))}..."
                        )

                if countdown_end is not None and now >= countdown_end:
                    frames = []
                    capture_deadline = now + CAPTURE_TIMEOUT_SECONDS
                    next_sample_at = now
                    countdown_end = None

                if capture_deadline is not None:
                    if now >= capture_deadline:
                        print(
                            f"Discarded {chr(selected_label + ord('A'))}: "
                            "not enough hand frames were detected."
                        )
                        selected_label = None
                        capture_deadline = None
                        frames = []
                    elif now >= next_sample_at:
                        next_sample_at = now + 1 / SAMPLE_RATE
                        if results.hand_landmarks:
                            frames.append(
                                landmarks_to_features(results.hand_landmarks[0])
                            )

                        if len(frames) == NUM_FRAMES:
                            sequence = add_velocity_features(
                                np.asarray(frames, dtype=np.float32),
                                sample_rate=SAMPLE_RATE,
                            )
                            row = [selected_label]
                            row.append(session_id)
                            row.extend(value for values in sequence for value in values)
                            with CSV_PATH.open(
                                mode="a", newline="", encoding="utf-8"
                            ) as csv_file:
                                csv.writer(csv_file).writerow(row)
                            letter = chr(selected_label + ord("A"))
                            print(
                                f"Saved {letter} sequence ({NUM_FRAMES} frames, "
                                f"session {session_id[:8]})."
                            )
                            selected_label = None
                            capture_deadline = None
                            frames = []

                if thumb_capture_deadline is not None:
                    if now >= thumb_capture_deadline:
                        if not thumb_frames:
                            print(
                                "No hand was detected; thumb example was not saved."
                            )
                        else:
                            capture_id = append_capture(
                                thumb_capture_label,
                                thumb_frames,
                                session_id,
                                THUMB_DATASET_PATH,
                            )
                            label_name = (
                                "thumbs-up"
                                if thumb_capture_label == THUMB_UP_LABEL
                                else "other pose"
                            )
                            print(
                                f"Saved {label_name} capture "
                                f"({len(thumb_frames)} frames, "
                                f"{capture_id[:8]})."
                            )
                            try:
                                counts = train_thumb_classifier()
                            except ValueError as error:
                                print(error)
                            else:
                                thumb_classifier = ThumbGestureClassifier()
                                print(
                                    "Trained thumb detector. Separate captures: "
                                    f"other pose={counts[OTHER_GESTURE_LABEL]}, "
                                    f"thumbs-up={counts[THUMB_UP_LABEL]}. "
                                    f"Saved {THUMB_MODEL_PATH.name}."
                                )
                        thumb_capture_label = None
                        thumb_capture_deadline = None
                        thumb_frames = []
                    elif now >= next_sample_at:
                        next_sample_at = now + 1 / SAMPLE_RATE
                        if results.hand_landmarks:
                            thumb_frames.append(
                                landmarks_to_features(
                                    results.hand_landmarks[0]
                                )[:POSE_FEATURES_PER_FRAME]
                            )

                if countdown_end is not None:
                    status = (
                        f"Get ready: {chr(selected_label + ord('A'))} "
                        f"{max(1, int(countdown_end - now + 0.99))}"
                    )
                elif capture_deadline is not None:
                    status = (
                        f"Recording {chr(selected_label + ord('A'))}: "
                        f"{len(frames)}/{NUM_FRAMES} frames"
                    )
                elif thumb_capture_deadline is not None:
                    label = (
                        "thumbs-up"
                        if thumb_capture_label == THUMB_UP_LABEL
                        else "other pose"
                    )
                    status = (
                        f"Recording {label}: {len(thumb_frames)} frames"
                    )
                else:
                    status = (
                        "A-Z: letter | Space: thumbs-up | "
                        "0: other pose | /: quit"
                    )

                cv2.putText(
                    frame,
                    status,
                    (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    status,
                    (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("ASL Gesture Sequence Collection", frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
