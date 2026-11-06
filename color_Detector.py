import cv2
import numpy as np
from sklearn.cluster import KMeans

# HSV color ranges: (lower HSV, upper HSV)
COLOR_RANGES = {
    "RED": [
        ((0, 110, 75), (10, 255, 255)),
        ((170, 110, 75), (179, 255, 255)),
    ],
    "ORANGE": [((11, 110, 75), (21, 255, 255))],
    "YELLOW": [((22, 100, 85), (35, 255, 255))],
    "GREEN": [((36, 65, 60), (85, 255, 255))],
    "BLUE": [((95, 85, 60), (130, 255, 255))],
    "PURPLE": [((131, 75, 60), (169, 255, 255))],
}

# Bounding box colors in BGR
BOX_COLORS = {
    "RED": (0, 0, 255),
    "ORANGE": (0, 165, 255),
    "YELLOW": (0, 255, 255),
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
    "PURPLE": (255, 0, 255),
}

MIN_AREA = 1000


def find_colors(frame):
    output = frame.copy()
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    detected = []
    kernel = np.ones((5, 5), dtype=np.uint8)

    for name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in ranges:
            lower_array = np.array(lower, dtype=np.uint8)
            upper_array = np.array(upper, dtype=np.uint8)
            current_mask = cv2.inRange(hsv, lower_array, upper_array)
            mask = cv2.bitwise_or(mask, current_mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            detected.append(name)
            color = BOX_COLORS[name]
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                output,                
                name,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

    return output, detected


def color_name_from_bgr(bgr):
    pixel = np.uint8([[bgr]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = map(int, hsv)

    if v < 55:
        return "BLACK"
    if s < 35 and v > 190:
        return "WHITE"
    if s < 45:
        return "GRAY"

    for name, ranges in COLOR_RANGES.items():
        for lower, upper in ranges:
            if all(lower[i] <= int(hsv[i]) <= upper[i] for i in range(3)):
                return name

    return "OTHER"


def dominant_color_kmeans(frame, roi):
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, "UNKNOWN"

    small = cv2.resize(crop, (64, 64))
    pixels = small.reshape(-1, 3).astype(np.float32)

    model = KMeans(n_clusters=3, random_state=42, n_init=3)
    labels = model.fit_predict(pixels)
    counts = np.bincount(labels, minlength=3)
    dominant_index = int(np.argmax(counts))
    dominant_bgr = model.cluster_centers_[dominant_index]
    dominant_bgr = np.clip(np.rint(dominant_bgr), 0, 255).astype(np.uint8)
    name = color_name_from_bgr(dominant_bgr)
    return dominant_bgr, name


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera neegdsen gui!")
        return

    print("Camera ajillaj baina")
    print("Q - Garah")
    frame_number = 0
    dominant_bgr = None
    dominant_name = "UNKNOWN"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cameraas zurag avch chadsangui")
                break

            frame = cv2.flip(frame, 1)
            result, detected = find_colors(frame)
            height, width = frame.shape[:2]
            side = min(height, width) // 3
            center_x = width // 2
            center_y = height // 2
            x1 = center_x - side // 2
            y1 = center_y - side // 2
            x2 = x1 + side
            y2 = y1 + side
            roi = (x1, y1, x2, y2)

            if frame_number % 12 == 0:
                dominant_bgr, dominant_name = dominant_color_kmeans(frame, roi)
            frame_number += 1

            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), 2)

            if dominant_bgr is not None:
                cv2.rectangle(
                    result,
                    (10, height - 60),
                    (65, height - 15),
                    tuple(int(c) for c in dominant_bgr),
                    -1,
                )
                cv2.putText(
                    result,
                    "Dominant: " + dominant_name,
                    (75, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.putText(
                result,
                "Objects: " + str(len(detected)),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Color Recognition System", result)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
