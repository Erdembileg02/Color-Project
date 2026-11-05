import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("camera opened")

while True:
    ret,frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask, mask2)
    
    
    cv2.imshow("camera", frame)
    cv2.imshow("Red Mask", red_mask )
    waitKey = cv2.waitKey(1)
    if  waitKey == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()