import cv2
import csv
import mediapipe as mp

CSV_FILE = 'asl_dataset.csv'

# Use the working import style from your webcam.py
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

print("Milestone 3: Strict ML Formatting Active!")
print("Hold a sign and press letters 'a' to 'z' to log samples (A=0, B=1...).")
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to find hands
    results = hands.process(rgb_frame)

    key = cv2.waitKey(1) & 0xFF

    # RULE 4: Failed Detections Filtered Out
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check if a letter key (a-z) was pressed
            if ord('a') <= key <= ord('z'):
                
                # RULE 2: Column 0 is the Integer Label (A=0, B=1, C=2...)
                integer_label = key - ord('a')
                row = [integer_label]

                # RULE 3: Columns 1 to 63 Order (x0, y0, z0, x1, y1, z1...)
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                # RULE 1: No Header Row (appending directly)
                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                
                print(f"Logged sample for '{chr(key).upper()}' as label [{integer_label}]")

    cv2.imshow('ASL Data Collection - Milestone 3', frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
