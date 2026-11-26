import time
from collections import deque

import cv2
import numpy as np

from custom_signs import (
    CUSTOM_SIGNS_PATH,
    CUSTOM_SIGN_MODEL_PATH,
    CustomSignClassifier,
    CustomSignLibrary,
    validate_sign_label,
)
from custom_train import train_and_save_custom_classifier
from hand_tracking import (
    create_hand_landmarker,
    detect_hand_landmarks,
    draw_hand_landmarks,
)
from predict import SignPredictor
from sequence_dataset import (
    NUM_FRAMES,
    POSE_FEATURES_PER_FRAME,
    add_velocity_features,
    landmarks_to_features,
)
from thumb_gesture import ThumbGestureClassifier
from word_prediction import (
    WORD_DECISION_SECONDS,
    WordPredictor,
)


STABLE_PREDICTION_SECONDS = 0.6
HAND_RELEASE_SECONDS = 0.35
SAMPLE_INTERVAL_SECONDS = 1 / 15
PREDICTION_SMOOTHING_WINDOWS = 5
SCROLL_SPEED_PIXELS_PER_SECOND = 120
CUSTOM_SIGN_RECORDINGS_PER_SESSION = 10
CUSTOM_SIGN_CAPTURE_TIMEOUT_SECONDS = 8
CUSTOM_SIGN_REST_SECONDS = 1.5
THUMBS_UP_HOLD_SECONDS = 0.35
NUMBER_SIGN_HOLD_SECONDS = 2.0
NUMBER_SIGN_LABELS = frozenset("0123456789")
DELETE_SIGN_HOLD_SECONDS = 3.0
DELETE_SIGN_LABEL = "delete"
DELETE_SIGN_LABELS = frozenset({"del", DELETE_SIGN_LABEL})
FONT = cv2.FONT_HERSHEY_SIMPLEX


class PredictionSmoother:
    def __init__(self, window_size):
        self.probability_history = deque(maxlen=window_size)

    def reset(self):
        self.probability_history.clear()

    def update(self, probabilities):
        self.probability_history.append(probabilities)
        return sum(self.probability_history) / len(self.probability_history)


class RecognizedSequence:
    def __init__(self):
        self.confirmed_words = []
        self.letters = []
        self.candidate_letter = None
        self.candidate_since = None
        self.last_committed_letter = None
        self.no_hand_since = None

    @property
    def text(self):
        items = list(self.confirmed_words)
        if self.current_word:
            items.append(self.current_word)
        return " ".join(items)

    @property
    def current_word(self):
        return "".join(self.letters).lower()

    def confirm_word(self, word):
        if not self.current_word or not word.startswith(self.current_word):
            raise ValueError(
                f"Confirmed word {word!r} does not match "
                f"the current prefix {self.current_word!r}."
            )
        self.confirmed_words.append(word)
        self.letters.clear()
        self.candidate_letter = None
        self.candidate_since = None
        self.last_committed_letter = None

    def keep_letters(self):
        if self.current_word:
            self.confirmed_words.append(self.current_word)
            self.letters.clear()
            if self.candidate_letter != DELETE_SIGN_LABEL:
                self.candidate_letter = None
                self.candidate_since = None
            self.last_committed_letter = None

    def delete_hold_remaining(self, now):
        if (
            self.candidate_letter != DELETE_SIGN_LABEL
            or self.candidate_since is None
        ):
            return None
        return max(
            0.0,
            DELETE_SIGN_HOLD_SECONDS - (now - self.candidate_since),
        )

    def delete_previous(self):
        if self.letters:
            self.letters.pop()
        elif self.confirmed_words:
            self.confirmed_words.pop()

    def commit(self, token):
        self.candidate_letter = None
        self.candidate_since = None
        if token.casefold() in DELETE_SIGN_LABELS:
            self.last_committed_letter = token
            self.delete_previous()
        elif len(token) == 1 and token.isdigit():
            self.keep_letters()
            self.confirmed_words.append(token)
            self.last_committed_letter = token
        else:
            self.letters.append(token)
            self.last_committed_letter = token
        return True

    def observe(self, letter, now):
        if letter is None:
            if self.no_hand_since is None:
                self.no_hand_since = now
            elif now - self.no_hand_since >= HAND_RELEASE_SECONDS:
                self.candidate_letter = None
                self.candidate_since = None
                self.last_committed_letter = None
            return False

        self.no_hand_since = None
        if letter.casefold() in DELETE_SIGN_LABELS:
            if self.candidate_letter != DELETE_SIGN_LABEL:
                self.candidate_letter = DELETE_SIGN_LABEL
                self.candidate_since = now
                return False
            if now - self.candidate_since >= DELETE_SIGN_HOLD_SECONDS:
                self.commit(DELETE_SIGN_LABEL)
                self.candidate_letter = DELETE_SIGN_LABEL
                self.candidate_since = now
                return True
            return False

        if letter == self.last_committed_letter:
            self.candidate_letter = None
            self.candidate_since = None
            return False

        if letter != self.candidate_letter:
            self.candidate_letter = letter
            self.candidate_since = now
            return False

        if now - self.candidate_since >= STABLE_PREDICTION_SECONDS:
            return self.commit(letter)
        return False


class NumberSignResolver:
    def __init__(self):
        self.pending_number = None
        self.pending_letter = None
        self.pending_since = None
        self.latched_number = None
        self.latched_letter = None

    def reset(self):
        self.pending_number = None
        self.pending_letter = None
        self.pending_since = None
        self.latched_number = None
        self.latched_letter = None

    def update(self, number, letter, now):
        resolved = []
        if self.latched_number is not None:
            if number == self.latched_number or (
                number is None and letter == self.latched_letter
            ):
                return resolved, self.latched_number, True
            self.latched_number = None
            self.latched_letter = None

        if self.pending_number is not None:
            if number == self.pending_number:
                if now - self.pending_since < NUMBER_SIGN_HOLD_SECONDS:
                    return resolved, self.pending_letter, True
                resolved.append(self.pending_number)
                self.latched_number = self.pending_number
                self.latched_letter = self.pending_letter
                self.pending_number = None
                self.pending_letter = None
                self.pending_since = None
                return resolved, number, True

            resolved.append(self.pending_letter)
            self.pending_number = None
            self.pending_letter = None
            self.pending_since = None

        if number is not None:
            self.pending_number = number
            self.pending_letter = letter
            self.pending_since = now
            return resolved, self.pending_letter, True

        return resolved, letter, False


def normalize_custom_sign_label(label):
    if label.casefold() in DELETE_SIGN_LABELS:
        return DELETE_SIGN_LABEL
    return label


def update_suggestion_after_commit(sequence, word_predictor, now):
    if not sequence.current_word:
        return None, None
    suggestions = word_predictor.suggest(sequence.current_word, limit=1)
    return (
        suggestions[0] if suggestions else None,
        now + WORD_DECISION_SECONDS,
    )


def is_thumbs_up(landmarks, classifier=None):
    if len(landmarks) != 21:
        return False

    if classifier is not None and classifier.model is not None:
        features = landmarks_to_features(landmarks)
        return classifier.is_thumbs_up(features[:POSE_FEATURES_PER_FRAME])

    thumb_is_raised = (
        landmarks[4].y + 0.02 < landmarks[3].y
        and landmarks[3].y < landmarks[2].y
    )
    other_fingers_are_folded = all(
        landmarks[tip].y > landmarks[pip].y
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
    )
    return thumb_is_raised and other_fingers_are_folded


def draw_word_suggestion(frame, suggestion, deadline, now):
    if deadline is None:
        return

    remaining = max(0.0, deadline - now)
    height, width = frame.shape[:2]
    panel_width = min(250, max(180, width // 3))
    panel_height = 94
    left = max(0, width - panel_width - 20)
    top = max(90, height // 3)
    cv2.rectangle(
        frame,
        (left, top),
        (left + panel_width, top + panel_height),
        (35, 35, 35),
        -1,
    )
    cv2.putText(
        frame, "Suggested word", (left + 12, top + 26), FONT, 0.55,
        (220, 220, 220), 1, cv2.LINE_AA,
    )
    word_text = suggestion if suggestion else "No match"
    cv2.putText(
        frame,
        word_text[:22],
        (left + 12, top + 57),
        FONT,
        0.85,
        (0, 255, 255) if suggestion else (150, 150, 150),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Thumbs-up or next letter ({remaining:.1f}s)",
        (left + 12, top + 82),
        FONT,
        0.36,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def draw_sign_sequence(frame, text, scroll_offset):
    height, width = frame.shape[:2]
    banner_top = max(0, height - 64)
    label = "Recognized:"
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
    word_predictor = WordPredictor()
    thumb_classifier = ThumbGestureClassifier()
    custom_signs = CustomSignLibrary()
    custom_classifier = CustomSignClassifier()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open webcam.")

    last_timestamp_ms = 0
    last_frame_time = time.monotonic()
    next_detection_at = 0.0
    scroll_offset = 0.0
    sequence = RecognizedSequence()
    active_suggestion = None
    decision_deadline = None
    thumbs_up_since = None
    thumbs_up_handled = False
    frame_buffer = deque(maxlen=NUM_FRAMES)
    prediction_smoother = PredictionSmoother(PREDICTION_SMOOTHING_WINDOWS)
    number_resolver = NumberSignResolver()
    current_letter = None
    current_confidence = 0.0
    current_is_custom = False
    recording_label = None
    recording_frames = []
    recording_count = 0
    recording_next_at = None
    recording_deadline = None
    results = None

    try:
        with create_hand_landmarker() as hand_landmarker:
            print(
                "Live 26-letter, number-sign, and custom-sign recognition active! "
                "Press 'q' to quit."
            )
            print(
                "Each recognized letter appears in the text buffer with one "
                "word suggestion. Hold thumbs-up to confirm, sign the next "
                "letter to update the suggestion, or wait 5 seconds to keep "
                "the letters as typed."
            )
            print(
                "Confirmed words are kept on screen and feedback is recorded "
                "in word_feedback.jsonl for later training."
            )
            print(
                "Using trained thumb gesture classifier."
                if thumb_classifier.model is not None
                else "No trained thumb model loaded; using the basic thumb pose "
                "detector. Collect examples with collect_data.py to train one."
            )
            print(
                f"Press 'c' to record a custom sign; its examples are saved to "
                f"{CUSTOM_SIGNS_PATH.name}."
            )
            print(
                "Enter a sign label in the terminal. Each recording session "
                f"captures {CUSTOM_SIGN_RECORDINGS_PER_SESSION} separate "
                f"{NUM_FRAMES}-frame examples, then trains the custom classifier."
            )
            print(
                "Record number signs with labels 0-9; each number prediction "
                f"requires a {NUMBER_SIGN_HOLD_SECONDS:.1f}-second hold. "
                f"Train a custom sign labeled {DELETE_SIGN_LABEL!r} "
                f"(or 'del') and hold it for "
                f"{DELETE_SIGN_HOLD_SECONDS:.1f} seconds per deletion."
            )
            if custom_classifier.model is None:
                print("No trained custom-sign classifier is currently loaded.")
            else:
                print(
                    f"Trained custom-sign classifier loaded from "
                    f"{CUSTOM_SIGN_MODEL_PATH.name}."
                )

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
                        sequence.no_hand_since = None
                        pose_features = landmarks_to_features(
                            results.hand_landmarks[0]
                        )
                        thumbs_up_detected = (
                            recording_label is None
                            and is_thumbs_up(
                                results.hand_landmarks[0], thumb_classifier
                            )
                        )
                        if not thumbs_up_detected:
                            frame_buffer.append(pose_features)

                        if (
                            recording_label is not None
                            and recording_next_at is not None
                            and now >= recording_next_at
                        ):
                            recording_frames.append(pose_features)
                            if len(recording_frames) == NUM_FRAMES:
                                custom_sequence = add_velocity_features(
                                    np.asarray(
                                        recording_frames, dtype=np.float32
                                    ),
                                )
                                custom_signs.add(recording_label, custom_sequence)
                                custom_classifier.model = None
                                recording_frames = []
                                recording_count += 1
                                if (
                                    recording_count
                                    == CUSTOM_SIGN_RECORDINGS_PER_SESSION
                                ):
                                    try:
                                        checkpoint, counts = (
                                            train_and_save_custom_classifier(
                                                custom_signs.signs
                                            )
                                        )
                                    except ValueError as error:
                                        print(
                                            "Examples were saved, but automatic "
                                            f"training failed: {error}"
                                        )
                                        print(
                                            "Run python custom_train.py after "
                                            "recording enough examples for every "
                                            "custom sign."
                                        )
                                    else:
                                        custom_classifier = CustomSignClassifier()
                                        print(
                                            f"Saved 10 examples of "
                                            f"{recording_label!r} and trained a "
                                            f"{checkpoint['model_type']} custom "
                                            f"classifier using "
                                            f"{sum(counts.values())} examples "
                                            f"across {len(counts)} signs."
                                        )
                                    recording_label = None
                                    recording_deadline = None
                                    recording_count = 0
                                    recording_next_at = None
                                else:
                                    print(
                                        f"Saved example {recording_count}/"
                                        f"{CUSTOM_SIGN_RECORDINGS_PER_SESSION} "
                                        f"for {recording_label!r}; reset your "
                                        "hand and repeat the sign."
                                    )
                                    recording_next_at = (
                                        now + CUSTOM_SIGN_REST_SECONDS
                                    )
                                    recording_deadline = (
                                        recording_next_at
                                        + CUSTOM_SIGN_CAPTURE_TIMEOUT_SECONDS
                                    )

                        if thumbs_up_detected:
                            resolved, _, _ = number_resolver.update(
                                None, None, now
                            )
                            for token in resolved:
                                sequence.commit(token)
                                active_suggestion, decision_deadline = (
                                    update_suggestion_after_commit(
                                        sequence, word_predictor, now
                                    )
                                )
                            number_resolver.reset()
                            frame_buffer.clear()
                            prediction_smoother.reset()
                            current_letter = None
                            current_confidence = 0.0
                            current_is_custom = False
                            sequence.observe(None, now)
                            if thumbs_up_since is None:
                                thumbs_up_since = now
                            elif (
                                not thumbs_up_handled
                                and now - thumbs_up_since
                                >= THUMBS_UP_HOLD_SECONDS
                            ):
                                if (
                                    active_suggestion is not None
                                    and decision_deadline is not None
                                    and now < decision_deadline
                                ):
                                    word_predictor.record_confirmation(
                                        sequence.current_word,
                                        [active_suggestion],
                                        active_suggestion,
                                    )
                                    sequence.confirm_word(active_suggestion)
                                    print(f"Confirmed word: {active_suggestion}")
                                    active_suggestion = None
                                    decision_deadline = None
                                thumbs_up_handled = True
                        else:
                            thumbs_up_since = None
                            thumbs_up_handled = False

                        if not thumbs_up_detected and len(frame_buffer) == NUM_FRAMES:
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
                            asl_letter = chr(ord("A") + predicted_index)
                            custom_prediction = None
                            if custom_classifier.model is not None:
                                custom_prediction = custom_classifier.predict(
                                    model_features
                                )
                            else:
                                custom_prediction = custom_signs.match(
                                    model_features
                                )
                            if custom_prediction in NUMBER_SIGN_LABELS:
                                (
                                    resolved_tokens,
                                    current_letter,
                                    number_is_pending,
                                ) = number_resolver.update(
                                    custom_prediction, asl_letter, now
                                )
                                current_is_custom = True
                            elif custom_prediction is not None:
                                resolved_tokens, _, _ = number_resolver.update(
                                    None, asl_letter, now
                                )
                                number_resolver.reset()
                                current_letter = normalize_custom_sign_label(
                                    custom_prediction
                                )
                                current_is_custom = True
                                number_is_pending = False
                            else:
                                (
                                    resolved_tokens,
                                    current_letter,
                                    number_is_pending,
                                ) = number_resolver.update(
                                    None, asl_letter, now
                                )
                                current_is_custom = False
                            for token in resolved_tokens:
                                sequence.commit(token)
                                if token.isdigit():
                                    active_suggestion = None
                                    decision_deadline = None
                                else:
                                    active_suggestion, decision_deadline = (
                                        update_suggestion_after_commit(
                                            sequence, word_predictor, now
                                        )
                                    )
                            if current_is_custom:
                                current_confidence = 0.0
                            else:
                                current_confidence = float(
                                    smoothed_probabilities[predicted_index] * 100
                                )
                            if (
                                not number_is_pending
                                and sequence.observe(current_letter, now)
                            ):
                                active_suggestion, decision_deadline = (
                                    update_suggestion_after_commit(
                                        sequence, word_predictor, now
                                    )
                                )
                    else:
                        sequence.observe(None, now)
                        if (
                            sequence.no_hand_since is not None
                            and now - sequence.no_hand_since
                            >= HAND_RELEASE_SECONDS
                        ):
                            resolved, _, _ = number_resolver.update(
                                None, None, now
                            )
                            for token in resolved:
                                sequence.commit(token)
                                if token.isdigit():
                                    active_suggestion = None
                                    decision_deadline = None
                                else:
                                    active_suggestion, decision_deadline = (
                                        update_suggestion_after_commit(
                                            sequence, word_predictor, now
                                        )
                                    )
                            number_resolver.reset()
                            frame_buffer.clear()
                            prediction_smoother.reset()
                            current_letter = None
                            current_confidence = 0.0
                            current_is_custom = False
                            thumbs_up_since = None
                            thumbs_up_handled = False

                if (
                    decision_deadline is not None
                    and now >= decision_deadline
                ):
                    sequence.keep_letters()
                    active_suggestion = None
                    decision_deadline = None

                if recording_deadline is not None and now >= recording_deadline:
                    print(
                        f"Stopped recording {recording_label!r} after "
                        f"{recording_count}/"
                        f"{CUSTOM_SIGN_RECORDINGS_PER_SESSION} examples. "
                        "Saved examples are kept; press 'c' again to continue."
                    )
                    recording_label = None
                    recording_frames = []
                    recording_count = 0
                    recording_next_at = None
                    recording_deadline = None

                if results is not None:
                    draw_hand_landmarks(frame, results)

                if recording_label is not None:
                    if recording_next_at is not None and now < recording_next_at:
                        display_text = (
                            f"Reset for {recording_label!r} "
                            f"({recording_count + 1}/"
                            f"{CUSTOM_SIGN_RECORDINGS_PER_SESSION})"
                        )
                    else:
                        display_text = (
                            f"Recording {recording_label!r} "
                            f"({recording_count + 1}/"
                            f"{CUSTOM_SIGN_RECORDINGS_PER_SESSION}): "
                            f"{len(recording_frames)}/{NUM_FRAMES} frames"
                        )
                    color = (0, 200, 255)
                elif results is None or not results.hand_landmarks:
                    display_text = "No hand detected"
                    color = (0, 165, 255)
                elif number_resolver.pending_number is not None:
                    remaining = max(
                        0.0,
                        NUMBER_SIGN_HOLD_SECONDS
                        - (now - number_resolver.pending_since),
                    )
                    display_text = (
                        f"Current: {current_letter} (hold for "
                        f"{number_resolver.pending_number}: "
                        f"{remaining:.1f}s)"
                    )
                    color = (0, 255, 0)
                elif current_letter == DELETE_SIGN_LABEL:
                    remaining = sequence.delete_hold_remaining(now)
                    if remaining is None:
                        display_text = "Delete sign detected"
                    else:
                        display_text = f"Hold to delete ({remaining:.1f}s)"
                    color = (0, 0, 255)
                elif sequence.current_word:
                    display_text = f"Spelling: {sequence.current_word}"
                    color = (0, 200, 255)
                elif current_letter is None:
                    display_text = (
                        f"Collecting gesture: {len(frame_buffer)}/"
                        f"{NUM_FRAMES} frames"
                    )
                    color = (0, 200, 255)
                else:
                    display_text = f"Current: {current_letter}"
                    if not current_is_custom:
                        display_text += f" ({current_confidence:.1f}%)"
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

                draw_word_suggestion(
                    frame, active_suggestion, decision_deadline, now
                )

                (label_width, _), _ = cv2.getTextSize(
                    "Recognized:", FONT, 0.8, 2
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
                draw_sign_sequence(frame, sequence.text, scroll_offset)

                cv2.imshow("ASL Gesture Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c") and recording_label is None:
                    try:
                        recording_label = validate_sign_label(
                            input("Enter the custom sign label: ")
                        )
                    except ValueError as error:
                        print(error)
                    else:
                        recording_frames = []
                        recording_count = 0
                        recording_next_at = (
                            time.monotonic() + CUSTOM_SIGN_REST_SECONDS
                        )
                        recording_deadline = (
                            recording_next_at
                            + CUSTOM_SIGN_CAPTURE_TIMEOUT_SECONDS
                        )
                        print(
                            f"Recording {recording_label!r}: "
                            f"{CUSTOM_SIGN_RECORDINGS_PER_SESSION} repetitions, "
                            f"{NUM_FRAMES} frames each. Perform each repetition "
                            "when the on-screen prompt says Recording; reset "
                            "your hand during the short pause between repetitions."
                        )
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
