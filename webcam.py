import time

import cv2

from predict import SignPredictor
from hand_tracking import (
    create_hand_landmarker,
    detect_hand_landmarks,
    draw_hand_landmarks,
)

STABLE_PREDICTION_SECONDS = 0.6
HAND_RELEASE_SECONDS = 0.35
SCROLL_SPEED_PIXELS_PER_SECOND = 120
FONT = cv2.FONT_HERSHEY_SIMPLEX


class LetterSequence:
    def __init__(self):
        self.letters = []
        self.candidate_letter = None
        self.candidate_since = None
        self.last_committed_letter = None
        self.no_hand_since = None

    @property
    def text(self):
        return " ".join(self.letters)

    def observe(self, letter, now):
        if letter is None:
            self.candidate_letter = None
            self.candidate_since = None
            if self.no_hand_since is None:
                self.no_hand_since = now
            elif now - self.no_hand_since >= HAND_RELEASE_SECONDS:
                self.last_committed_letter = None
            return

        self.no_hand_since = None
        if letter == self.last_committed_letter:
            self.candidate_letter = None
            self.candidate_since = None
            return

        if letter != self.candidate_letter:
            self.candidate_letter = letter
            self.candidate_since = now
            return

        if now - self.candidate_since >= STABLE_PREDICTION_SECONDS:
            self.letters.append(letter)
            self.last_committed_letter = letter
            self.candidate_letter = None
            self.candidate_since = None


def draw_letter_sequence(frame, text, scroll_offset):
    height, width = frame.shape[:2]
    banner_top = max(0, height - 64)
    label = "Letters:"
    label_x = 16
    baseline_y = height - 20
    font_scale = 0.8
    thickness = 2
    (label_width, _), _ = cv2.getTextSize(label, FONT, font_scale, thickness)
    text_left = label_x + label_width + 16

    cv2.rectangle(frame, (0, banner_top), (width, height), (25, 25, 25), -1)
    cv2.putText(
        frame,
        label,
        (label_x, baseline_y),
        FONT,
        font_scale,
        (220, 220, 220),
        thickness,
        cv2.LINE_AA,
    )

    text_region = frame[banner_top:height, text_left:width]
    if text and text_region.size:
        cv2.putText(
            text_region,
            text,
            (10 - round(scroll_offset), baseline_y - banner_top),
            FONT,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def main():
    predictor = SignPredictor()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open webcam.")

    last_timestamp_ms = 0
    last_frame_time = time.monotonic()
    scroll_offset = 0.0
    sequence = LetterSequence()
    try:
        with create_hand_landmarker() as hand_landmarker:
            print("Live ASL letter prediction active! Press 'q' to quit.")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                now = time.monotonic()
                frame_elapsed = now - last_frame_time
                last_frame_time = now
                timestamp_ms = max(
                    time.monotonic_ns() // 1_000_000,
                    last_timestamp_ms + 1,
                )
                last_timestamp_ms = timestamp_ms
                results = detect_hand_landmarks(hand_landmarker, frame, timestamp_ms)
                draw_hand_landmarks(frame, results)

                if results.hand_landmarks:
                    landmarks = results.hand_landmarks[0]
                    coordinates = [
                        coordinate
                        for landmark in landmarks
                        for coordinate in (landmark.x, landmark.y, landmark.z)
                    ]
                    letter, confidence = predictor.predict(coordinates)
                    sequence.observe(letter, now)
                    display_text = f"Current: {letter} ({confidence:.1f}%)"
                    color = (0, 255, 0)
                else:
                    sequence.observe(None, now)
                    display_text = "No hand detected"
                    color = (0, 165, 255)

                cv2.putText(
                    frame,
                    display_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    display_text,
                    (20, 40),
                    FONT,
                    1,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                label_width, _ = cv2.getTextSize(
                    "Letters:", FONT, 0.8, 2
                )[0]
                text_region_width = max(0, frame.shape[1] - (16 + label_width + 16))
                (sequence_width, _), _ = cv2.getTextSize(
                    sequence.text, FONT, 0.8, 2
                )
                max_scroll = max(0, sequence_width - max(0, text_region_width - 20))
                scroll_offset = min(
                    max_scroll,
                    scroll_offset + SCROLL_SPEED_PIXELS_PER_SECOND * frame_elapsed,
                )
                draw_letter_sequence(frame, sequence.text, scroll_offset)

                cv2.imshow("ASL Recognizer - Milestone 2", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
