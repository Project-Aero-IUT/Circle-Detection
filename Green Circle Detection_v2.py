import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

prev_center = None
prev_radius = None
radius_threshold = 2  # adjust for sensitivity
frame_count = 0
accurate_detections = 0
start_time = time.time()

frame_counter = 0
frame_interval = 5  # Check direction every 5 frames
last_direction = "Searching..."
threshold = 5  # pixel threshold for movement

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_counter += 1

    frame_time_start = time.time()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    output = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=50, param2=30, minRadius=10, maxRadius=100)

    if circles is not None:
        accurate_detections += 1
        circles = np.uint16(np.around(circles))
        

        for i in circles[0, :1]:
            center = (int(i[0]), int(i[1]))  # Convert to int to avoid overflow
            radius = int(i[2])
            cv2.circle(frame, center, 1, (0, 100, 100), 3)
            cv2.circle(frame, center, radius, (255, 0, 255), 3)

            if prev_center and prev_radius and frame_counter >= frame_interval:
                dx = center[0] - prev_center[0]
                dy = center[1] - prev_center[1]
                dr = radius - prev_radius
            
                movement_directions = []

                if abs(dx) > threshold:
                    movement_directions.append("go right" if dx > 0 else "go left")

                if abs(dy) > threshold:
                    movement_directions.append("go down" if dy > 0 else "go up")

                if abs(dr) > radius_threshold:
                    movement_directions.append("go forward" if dr > 0 else "go backward")
                    print(f"Radius: {radius}, Prev: {prev_radius}, Δr = {radius - prev_radius}")
                    cv2.putText(frame, f"Radius: {radius}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

                if movement_directions:
                    last_direction = movement_directions[0]  # Pick the first one
                else:
                    last_direction = "stay"


                frame_counter = 0  # Reset counter

                # Debug print:
                #print(f"dx: {dx}, dy: {dy}, dr: {dr}, direction: {last_direction}")

            prev_center = center
            prev_radius = radius

    else:
        last_direction = "No circle detected"

    delay = time.time() - frame_time_start

    # Display text on frame
    cv2.putText(frame, f"Direction: {last_direction}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Delay: {delay:.3f}s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    


    cv2.imshow("Tracking Green Circle", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
cap.release()
cv2.destroyAllWindows()

accuracy = (accurate_detections / frame_count) * 100
print(f"Average FPS: {frame_count / (end_time - start_time):.2f}")
print(f"Detection Accuracy: {accuracy:.2f}%")
