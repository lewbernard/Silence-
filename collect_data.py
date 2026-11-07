import csv
import time
from pathlib import Path

import cv2

from hand_tracking import (
    create_hand_landmarker,
    detect_hand_landmarks,
    draw_hand_landmarks,
)
from sequence_dataset import NUM_FRAMES, SEQUENCE_CSV, landmarks_to_features


CSV_PATH = Path(__file__).resolve().with_name(SEQUENCE_CSV)
SAMPLE_RATE = 15
COUNTDOWN_SECONDS = 0
CAPTURE_TIMEOUT_SECONDS = 8


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open webcam.")

    last_timestamp_ms = 0
    selected_label = None
    countdown_end = None
    capture_deadline = None
    next_sample_at = 0.0
    frames = []

    try:
        with create_hand_landmarker() as hand_landmarker:
            print(f"Recording gesture sequences to {CSV_PATH.name}.")
            print(
                "Press A-Z to record one example. Hold static letters; "
                "draw the full motion for moving letters."
            )
            print("Record examples for every letter; press '/' to quit.")

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

                if countdown_end is None and capture_deadline is None:
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
                            row = [selected_label]
                            row.extend(value for values in frames for value in values)
                            with CSV_PATH.open(
                                mode="a", newline="", encoding="utf-8"
                            ) as csv_file:
                                csv.writer(csv_file).writerow(row)
                            letter = chr(selected_label + ord("A"))
                            print(f"Saved {letter} sequence ({NUM_FRAMES} frames).")
                            selected_label = None
                            capture_deadline = None
                            frames = []

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
                else:
                    status = "Press A-Z to record | q to quit"

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
