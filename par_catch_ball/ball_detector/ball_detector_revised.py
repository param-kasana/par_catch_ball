import cv2
import numpy as np

# Calibrated HSV range for your rubber ball
lower_orange = np.array([2, 150, 180])
upper_orange = np.array([10, 255, 255])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame")
        break

    # Step 1: Convert to HSV and mask orange color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Step 2: Preprocess mask for circle detection
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)

    # Step 3: Apply Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=100, param2=15, minRadius=10, maxRadius=60)

    # Step 4: If a circle is found, draw it
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :1]:  # only the first detected circle
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.putText(frame, f"Ball @ ({x}, {y})", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"✅ Detected circle: x={x}, y={y}, radius={r}")
            break

    # Show result
    cv2.imshow("Ball Detection", frame)
    cv2.imshow("HSV Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
