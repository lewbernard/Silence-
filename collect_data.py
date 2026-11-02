import cv2
import csv
import os
import mediapipe.solutions.hands as mp_hands
import mediapipe.solutions.drawing_utils as mp_drawing

CSV_FILE = 'asl_dataset.csv'

# Initialize CSV header if the file does not exist yet
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

print("Milestone 3 Active!")
print("Hold an ASL sign in front of the camera and press 'a', 'b', or 'c' to log a snapshot.")
print("Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    key = cv2.waitKey(1) & 0xFF

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Capture coordinates when 'a', 'b', or 'c' is pressed
            if key in [ord('a'), ord('b'), ord('c')]:
                label = chr(key).upper()
                row = [label]

                # Extract all 21 (x, y, z) landmarks into a flat list of 63 numbers
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                
                print(f"Recorded sample for letter: {label}")

    cv2.imshow('ASL Data Collection - Milestone 3', frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()