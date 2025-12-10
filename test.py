import cv2
import mediapipe as mp
import pickle

model = pickle.load(open("rps_model4.pkl", "rb"))

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                row = []
                for landmark in lm.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z])

                prediction = model.predict([row])[0]

                cv2.putText(frame, prediction, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
