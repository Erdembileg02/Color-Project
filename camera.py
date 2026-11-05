
import cv2

# Камер нээх (0 = үндсэн камер)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Камер нээгдсэнгүй!")
    exit()

while True:
    # Камераас нэг кадр унших
    ret, frame = cap.read()

    if not ret:
        print("Дүрс уншиж чадсангүй!")
        break

    # Камерын дүрсийг харуулах
    cv2.imshow("My Camera", frame)

    # Q товч дарвал зогсоох
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Камерыг чөлөөлөх
cap.release()
cv2.destroyAllWindows()
