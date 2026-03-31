from picamera2 import Picamera2
import cv2
import numpy as np

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={'size': (640, 480), 'format': 'RGB888'}))
cam.start()

while True:
    frame = cam.capture_array()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Higher saturation (150) and value (100) thresholds filter out skin tones
    mask1 = cv2.inRange(hsv, np.array([0, 150, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 150, 100]), np.array([180, 255, 255]))
    mask = mask1 | mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > 200:  # bigger minimum — filters small skin patches
            x, y, w, h = cv2.boundingRect(largest)

            # Aspect ratio filter — crab is roughly squarish, not a thin sliver
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.3 < aspect_ratio < 3.0:

                # Fill ratio — at least 30% of bounding box must be red
                box_area = w * h
                red_in_box = cv2.countNonZero(mask[y:y+h, x:x+w])
                fill_ratio = red_in_box / box_area if box_area > 0 else 0
                if fill_ratio > 0.3:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"RESCUE SUBJECT area:{int(area)}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Detection Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
