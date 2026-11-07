import csv
import time
from pathlib import Path

import cv2

from hand_tracking import (
    create_hand_landmarker,
    detect_hand_landmarks,
    draw_hand_landmarks,
    landmarks_to_csv_row,
)


CSV_FILE = Path(__file__).resolve().with_name("asl_dataset.csv")


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open webcam.")

    last_timestamp_ms = 0
    try:
        with create_hand_landmarker() as hand_landmarker:
            print("Milestone 3: Strict ML Formatting Active!")
            print("Hold a sign and press letters 'a' to 'z' to log samples (A=0, B=1...).")
            print("Press '/' to quit.")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                timestamp_ms = max(
                    time.monotonic_ns() // 1_000_000,
                    last_timestamp_ms + 1,
                )
                last_timestamp_ms = timestamp_ms
                results = detect_hand_landmarks(hand_landmarker, frame, timestamp_ms)
                key = cv2.waitKey(1) & 0xFF

                if results.hand_landmarks:
                    draw_hand_landmarks(frame, results)

                    if ord("a") <= key <= ord("z"):
                        integer_label = key - ord("a")
                        row = landmarks_to_csv_row(
                            integer_label, results.hand_landmarks[0]
                        )

                        with CSV_FILE.open(mode="a", newline="") as csv_file:
                            csv.writer(csv_file).writerow(row)

                        print(
                            f"Logged sample for '{chr(key).upper()}' "
                            f"as label [{integer_label}]"
                        )

                cv2.imshow("ASL Data Collection - Milestone 3", frame)

                if key == ord("/"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
