import cv2
import numpy as np
import torch
import time
import csv

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture(0)

prev_center = None
prev_radius = None

# Metrics
frame_times = []
circle_detected_log = []
direction_log = []

total_frames = 0
circle_detected_frames = 0

def detect_green_circle(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    blurred = cv2.GaussianBlur(mask, (7, 7), 0)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=30, minRadius=5, maxRadius=100
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles[0]  # Return the first detected circle
    return None

def get_direction(prev, curr, prev_r, curr_r):
    if prev is None:
        return "Still"

    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    dr = curr_r - prev_r

    pos_thresh = 10
    radius_thresh = 3

    dir_x = ""
    dir_y = ""
    dir_z = ""

    if dx > pos_thresh:
        dir_x = "Right"
    elif dx < -pos_thresh:
        dir_x = "Left"

    if dy > pos_thresh:
        dir_y = "Down"
    elif dy < -pos_thresh:
        dir_y = "Up"

    if dr > radius_thresh:
        dir_z = "Forward"
    elif dr < -radius_thresh:
        dir_z = "Backward"

    directions = [d for d in [dir_z, dir_y, dir_x] if d]
    return " ".join(directions) if directions else "Still"

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1

    # Optional: YOLO detections
    results = model(frame)
    df = results.pandas().xyxy[0]
    for _, row in df.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Detect green circle
    circle = detect_green_circle(frame)
    direction = "Unknown"
    if circle is not None:
        cx, cy, r = circle
        green_circle_center = (cx, cy)

        # Draw circle
        cv2.circle(frame, green_circle_center, r, (0, 255, 0), 2)
        cv2.putText(frame, "Green Circle", (cx - 20, cy - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        direction = get_direction(prev_center, green_circle_center, prev_radius or r, r)

        prev_center = green_circle_center
        prev_radius = r

        circle_detected_frames += 1
        circle_detected_log.append(1)
    else:
        prev_center = None
        prev_radius = None
        circle_detected_log.append(0)

    direction_log.append(direction)

    cv2.putText(frame, f"Move: {direction}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Measure frame delay
    end_time = time.time()
    frame_delay = (end_time - start_time) * 1000
    frame_times.append(frame_delay)

    # Show delay and FPS
    fps = 1000 / frame_delay if frame_delay > 0 else 0
    cv2.putText(frame, f"Delay: {frame_delay:.2f} ms", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("YOLO + Green Circle Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Summary statistics
avg_delay = sum(frame_times) / len(frame_times)
accuracy = (circle_detected_frames / total_frames) * 100

print("\n--- Summary ---")
print(f"Total frames: {total_frames}")
print(f"Circle detected: {circle_detected_frames}")
print(f"Detection accuracy: {accuracy:.2f}%")
print(f"Average frame delay: {avg_delay:.2f} ms")

# Save CSV
with open("frame_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Frame", "Delay_ms", "FPS", "Direction", "CircleDetected"])
    for i in range(total_frames):
        fps_val = 1000 / frame_times[i] if frame_times[i] > 0 else 0
        writer.writerow([
            i+1,
            f"{frame_times[i]:.2f}",
            f"{fps_val:.2f}",
            direction_log[i],
            circle_detected_log[i]
        ])

print("✅ Frame data saved to 'frame_metrics.csv'")
