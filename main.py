import cv2
import mediapipe as mp
import csv
import sys

LABEL = "paper"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Temporary storage (not saved until confirmed)
temp_data = []

clicked = False

def on_mouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x <= 150 and 10 <= y <= 60:
            clicked = True

cv2.namedWindow("Dataset Collector")
cv2.setMouseCallback("Dataset Collector", on_mouse)

try:
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while True:

            if clicked:
                break

            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # STOP button
            cv2.rectangle(frame, (10, 10), (150, 60), (0, 0, 255), -1)
            cv2.putText(frame, "STOP", (35, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.putText(frame, f"Capturing: {LABEL}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Save landmarks into memory
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                    row = []
                    for landmark in lm.landmark:
                        row.extend([landmark.x, landmark.y, landmark.z])

                    row.append(LABEL)
                    temp_data.append(row)

            cv2.imshow("Dataset Collector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\n\n⛔ CTRL + C detected — Data NOT saved.")
    temp_data = []  # discard all
    pass

cap.release()
cv2.destroyAllWindows()

# ===============================
# CONFIRMATION BEFORE SAVING
# ===============================

if len(temp_data) == 0:
    print("No data captured. Exiting.")
    sys.exit()

print("\nYou captured", len(temp_data), "samples.")
choice = input("Save data to CSV? (y/n): ").lower()

if choice == "y":
    with open("hand_data.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(temp_data)
    print("✔ Data saved successfully to hand_data.csv")
else:
    print("❌ Data NOT saved.")

print("Done.")
