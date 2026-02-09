import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Barmoq uchlari indexlari
finger_tips = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    finger_count = 0

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark

        # 🔹 BOSH BARMOQ (x bo‘yicha tekshiramiz)
        if lm[finger_tips[0]].x > lm[finger_tips[0] - 1].x:
            finger_count += 1

        # 🔹 QOLGAN 4 TA BARMOQ (y bo‘yicha)
        for i in range(1, 5):
            if lm[finger_tips[i]].y < lm[finger_tips[i] - 2].y:
                finger_count += 1

        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 🔢 Natijani chiqarish
    cv2.putText(
        img,
        f'Finger Count: {finger_count}',
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        4
    )

    cv2.imshow("Finger Counter - Step 2", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()