import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
finger_tips = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    total_finger_count = 0

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            lm = hand_landmarks.landmark
            finger_count = 0

            # Qo'l labelini aniqlash (chap/ o'ng)
            hand_label = results.multi_handedness[i].classification[0].label

            # 🔹 BOSH BARMOQ
            if hand_label == "Right":  # o'ng qo'l
                if lm[finger_tips[0]].x > lm[finger_tips[0] - 1].x:
                    finger_count += 1
            else:  # chap qo'l
                if lm[finger_tips[0]].x < lm[finger_tips[0] - 1].x:
                    finger_count += 1

            # 🔹 QOLGAN 4 TA BARMOQ
            for j in range(1, 5):
                if lm[finger_tips[j]].y < lm[finger_tips[j] - 2].y:
                    finger_count += 1

            total_finger_count += finger_count

            # Qo'l landmarklarini chizish
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(
        img,
        f'Total Fingers: {total_finger_count}',
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        4
    )

    cv2.imshow("Finger Counter - 2 Hands", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()