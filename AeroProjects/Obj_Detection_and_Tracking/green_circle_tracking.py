import cv2
import numpy as np
import time
import csv
from datetime import datetime

lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

cap = cv2.VideoCapture(0)

prev_center = None
prev_radius = None
prev_time = time.time()

kernel = np.ones((5, 5), np.uint8)

movement_threshold = 7
radius_threshold = 3


movement_file = open('movements.csv', mode='w', newline='')
movement_writer = csv.writer(movement_file)
movement_writer.writerow(['Timestamp_UTC', 'Timestamp_Readable', 'Direction'])

frame_file = open('frame_stats.csv', mode='w', newline='')
frame_writer = csv.writer(frame_file)
frame_writer.writerow(['Timestamp_UTC', 'Timestamp_Readable', 'Frame Delay (s)', 'FPS', 'Accuracy'])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    direction = "Still"
    accuracy = 0.0  # circularity default

    if contours:
        circular_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > 0.6 and area > 300:  # loosened threshold
                circular_contours.append((cnt, circularity))

        if circular_contours:
            largest_cnt, accuracy = max(circular_contours, key=lambda x: cv2.contourArea(x[0]))
            area = cv2.contourArea(largest_cnt)

            ((x, y), radius) = cv2.minEnclosingCircle(largest_cnt)

            if radius > 10:
                center = (int(x), int(y))
                cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
                cv2.putText(frame, "Green Circle", (center[0] - 50, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if prev_center is not None:
                    dx = center[0] - prev_center[0]
                    dy = center[1] - prev_center[1]

                    if abs(dx) > abs(dy):
                        if dx > movement_threshold:
                            direction = "Right"
                        elif dx < -movement_threshold:
                            direction = "Left"
                    else:
                        if dy > movement_threshold:
                            direction = "Down"
                        elif dy < -movement_threshold:
                            direction = "Up"

                    if prev_radius is not None:
                        dr = radius - prev_radius
                        if abs(dr) > radius_threshold:
                            if dr > 0:
                                direction = "Forward"
                            else:
                                direction = "Backward"

                prev_center = center
                prev_radius = radius

    curr_time = time.time()
    frame_delay = curr_time - prev_time
    fps = 1 / frame_delay if frame_delay > 0 else 0
    prev_time = curr_time

   
    timestamp_readable = datetime.now().strftime('%H:%M:%S.%f')[:-3]

    movement_writer.writerow([curr_time, timestamp_readable, direction])
    frame_writer.writerow([curr_time, timestamp_readable, f"{frame_delay:.6f}", f"{fps:.2f}", f"{accuracy:.4f}"])

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Direction: {direction}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Green Circle Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

movement_file.close()
frame_file.close()
