
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera neegdsenгүй!")
    exit()

print("Camera opened")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Камерын баруун, зүүн талыг солих
    frame = cv2.flip(frame, 1)

    # BGR-ээс HSV рүү хувиргах
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Улаан өнгийн хоёр хязгаар
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([179, 255, 255])

    # Улаан өнгийг ялгах
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask, mask2)

    # Улаан өнгийн объектуудын хүрээг олох
    contours, hierarchy = cv2.findContours(
        red_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)

        # Хэт жижиг хэсгүүдийг алгасах
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)

            # Ногоон дөрвөлжин хүрээ
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "RED",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("Camera", frame)
    cv2.imshow("Red Mask", red_mask)

    # Q товч дарахад хаах
    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
