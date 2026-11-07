import time
from collections import deque

import cv2

from hand_tracking import (
    create_hand_landmarker,
    detect_hand_landmarks,
    draw_hand_landmarks,
)
from predict import SignPredictor
from sequence_dataset import (
    NUM_FRAMES,
    add_velocity_features,
    landmarks_to_features,
)


STABLE_PREDICTION_SECONDS = 0.6
HAND_RELEASE_SECONDS = 0.35
SAMPLE_INTERVAL_SECONDS = 1 / 15
PREDICTION_SMOOTHING_WINDOWS = 5
SCROLL_SPEED_PIXELS_PER_SECOND = 120
FONT = cv2.FONT_HERSHEY_SIMPLEX


class PredictionSmoother:
    def __init__(self, window_size):
        self.probability_history = deque(maxlen=window_size)

    def reset(self):
        self.probability_history.clear()

    def update(self, probabilities):
        self.probability_history.append(probabilities)
        return sum(self.probability_history) / len(self.probability_history)


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
    next_detection_at = 0.0
    scroll_offset = 0.0
    sequence = LetterSequence()
    frame_buffer = deque(maxlen=NUM_FRAMES)
    prediction_smoother = PredictionSmoother(PREDICTION_SMOOTHING_WINDOWS)
    current_letter = None
    current_confidence = 0.0
    results = None

    try:
        with create_hand_landmarker() as hand_landmarker:
            print("Live 26-letter gesture recognition active! Press 'q' to quit.")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                now = time.monotonic()
                elapsed = now - last_frame_time
                last_frame_time = now
                if now >= next_detection_at:
                    timestamp_ms = max(
                        time.monotonic_ns() // 1_000_000,
                        last_timestamp_ms + 1,
                    )
                    last_timestamp_ms = timestamp_ms
                    results = detect_hand_landmarks(
                        hand_landmarker, frame, timestamp_ms
                    )
                    next_detection_at = now + SAMPLE_INTERVAL_SECONDS

                    if results.hand_landmarks:
                        frame_buffer.append(
                            landmarks_to_features(results.hand_landmarks[0])
                        )

                        if len(frame_buffer) == NUM_FRAMES:
                            model_features = add_velocity_features(
                                list(frame_buffer)
                            )
                            probabilities = predictor.predict_probabilities(
                                model_features
                            )
                            smoothed_probabilities = prediction_smoother.update(
                                probabilities
                            )
                            predicted_index = int(smoothed_probabilities.argmax())
                            current_letter = chr(ord("A") + predicted_index)
                            current_confidence = float(
                                smoothed_probabilities[predicted_index] * 100
                            )
                            sequence.observe(current_letter, now)
                    else:
                        frame_buffer.clear()
                        prediction_smoother.reset()
                        current_letter = None
                        current_confidence = 0.0
                        sequence.observe(None, now)

                if results is not None:
                    draw_hand_landmarks(frame, results)

                if results is None or not results.hand_landmarks:
                    display_text = "No hand detected"
                    color = (0, 165, 255)
                elif current_letter is None:
                    display_text = (
                        f"Collecting gesture: {len(frame_buffer)}/{NUM_FRAMES} "
                        "frames"
                    )
                    color = (0, 200, 255)
                else:
                    display_text = (
                        f"Current: {current_letter} "
                        f"({current_confidence:.1f}%)"
                    )
                    color = (0, 255, 0)

                cv2.putText(
                    frame,
                    display_text,
                    (20, 40),
                    FONT,
                    0.85,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    display_text,
                    (20, 40),
                    FONT,
                    0.85,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                (label_width, _), _ = cv2.getTextSize(
                    "Letters:", FONT, 0.8, 2
                )
                text_region_width = max(
                    0, frame.shape[1] - (16 + label_width + 16)
                )
                (sequence_width, _), _ = cv2.getTextSize(
                    sequence.text, FONT, 0.8, 2
                )
                max_scroll = max(
                    0, sequence_width - max(0, text_region_width - 20)
                )
                scroll_offset = min(
                    max_scroll,
                    scroll_offset + SCROLL_SPEED_PIXELS_PER_SECOND * elapsed,
                )
                draw_letter_sequence(frame, sequence.text, scroll_offset)

                cv2.imshow("ASL Gesture Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
