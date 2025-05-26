import cv2
import numpy as np

# Adjusted HSV range
lower_orange = np.array([2, 150, 180])
upper_orange = np.array([10, 255, 255])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:  # skip tiny blobs
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)

            if radius > 5:
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                cv2.putText(frame, f"Ball @ {center}", (center[0]+10, center[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                break  # Only highlight one ball

    # Show original and mask
    cv2.imshow("Ball Detection", frame)
    cv2.imshow("HSV Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
