# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node

# from geometry_msgs.msg import PointStamped
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import numpy as np
# import cv2


# class KalmanFilter2D:
#     def __init__(self, dt=1.0):
#         # State: [x, y, vx, vy]
#         self.x = np.zeros((4, 1))
#         self.F = np.array([
#             [1, 0, dt, 0],
#             [0, 1, 0, dt],
#             [0, 0, 1,  0],
#             [0, 0, 0,  1]
#         ])
#         self.H = np.array([
#             [1, 0, 0, 0],
#             [0, 1, 0, 0]
#         ])
#         self.R = np.eye(2) * 30.0     # measurement noise
#         self.Q = np.eye(4) * 1.0      # process noise
#         self.P = np.eye(4) * 500.0    # error covariance
#         self.initialized = False

#     def init_state(self, x, y):
#         self.x = np.array([[x], [y], [0], [0]])
#         self.initialized = True

#     def predict(self):
#         self.x = self.F @ self.x
#         self.P = self.F @ self.P @ self.F.T + self.Q
#         return self.x[0, 0], self.x[1, 0]

#     def update(self, z):
#         z = np.array(z).reshape(2, 1)
#         y = z - (self.H @ self.x)
#         S = self.H @ self.P @ self.H.T + self.R
#         K = self.P @ self.H.T @ np.linalg.inv(S)
#         self.x = self.x + K @ y
#         self.P = (np.eye(4) - K @ self.H) @ self.P


# class TrajectoryEstimator(Node):
#     def __init__(self):
#         super().__init__('trajectory_estimator')
#         self.bridge = CvBridge()
#         self.kalman = KalmanFilter2D()
#         self.filtered_points = []

#         self.latest_header = None

#         self.sub_point = self.create_subscription(
#             PointStamped,
#             '/ball_point',
#             self.point_callback,
#             10
#         )
#         self.sub_image = self.create_subscription(
#             Image,
#             '/image_topic',
#             self.image_callback,
#             10
#         )
#         self.pub_image = self.create_publisher(
#             Image,
#             '/trajectory_image',
#             10
#         )

#     def point_callback(self, msg: PointStamped):
#         x, y = msg.point.x, msg.point.y
#         self.latest_header = msg.header

#         if not self.kalman.initialized:
#             self.kalman.init_state(x, y)
#         else:
#             self.kalman.update([x, y])
#             self.kalman.predict()

#         filtered_x, filtered_y = self.kalman.x[0, 0], self.kalman.x[1, 0]
#         self.filtered_points.append((int(filtered_x), int(filtered_y)))
#         if len(self.filtered_points) > 50:
#             self.filtered_points.pop(0)

#     def image_callback(self, msg: Image):
#         try:
#             frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#         except Exception as e:
#             self.get_logger().error(f"Image conversion failed: {e}")
#             return

#         # # Draw past (filtered) trajectory
#         # if len(self.filtered_points) >= 2:
#         #     for i in range(1, len(self.filtered_points)):
#         #         cv2.line(
#         #             frame,
#         #             self.filtered_points[i - 1],
#         #             self.filtered_points[i],
#         #             (0, 255, 255), 2  # yellow
#         #         )

#         # Draw future predicted trajectory
#         if self.kalman.initialized:
#             pred_kf = KalmanFilter2D()
#             pred_kf.x = self.kalman.x.copy()
#             pred_kf.P = self.kalman.P.copy()
#             pred_pts = []

#             for _ in range(60):  # simulate 30 future steps
#                 px, py = pred_kf.predict()
#                 pred_pts.append((int(px), int(py)))

#             for i in range(1, len(pred_pts)):
#                 cv2.line(
#                     frame,
#                     pred_pts[i - 1],
#                     pred_pts[i],
#                     (255, 0, 0), 2  # blue
#                 )

#         # Publish the annotated image
#         try:
#             debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
#             if self.latest_header:
#                 debug_msg.header = self.latest_header
#             self.pub_image.publish(debug_msg)
#         except Exception as e:
#             self.get_logger().error(f"Failed to publish debug image: {e}")


# def main(args=None):
#     rclpy.init(args=args)
#     node = TrajectoryEstimator()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()






#-----------------------------------------------------------------






#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from collections import deque

# === State machine enums ===
STATE_SET = 0
STATE_STANDBY = 1
STATE_FLYING = 2
STATE_INVISIBLE = 3
STATE_NOT_FLYING = 4
STATE_NAMES = ["SET", "STANDBY", "FLYING", "INVISIBLE", "NOT FLYING"]

class KalmanFilter2D:
    def __init__(self, dt=1.0):
        self.x = np.zeros((4, 1))
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.R = np.eye(2) * 30.0
        self.Q = np.eye(4) * 1.0
        self.P = np.eye(4) * 500.0
        self.initialized = False

    def init_state(self, x, y):
        self.x = np.array([[x], [y], [0], [0]])
        self.initialized = True

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0, 0], self.x[1, 0]

    def update(self, z):
        z = np.array(z).reshape(2, 1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

class TrajectoryEstimator(Node):
    def __init__(self):
        super().__init__('trajectory_estimator')
        self.bridge = CvBridge()
        self.kalman = KalmanFilter2D()
        self.state = STATE_SET
        self.state_str = STATE_NAMES[self.state]
        self.missing_count = 0
        self.kf_history = deque(maxlen=50)
        self.time_history = deque(maxlen=50)
        self.latest_header = None
        self.dt = 1.0  # Default; updated from timestamps
        self.invisible_limit = 6  # Number of missed frames before "NOT FLYING"
        self.flying_dist_thresh = 8.0  # Minimum pixel jump to consider "flying"
        self.image_width = 640  # Update as per your image
        self.image_height = 480

        self.sub_point = self.create_subscription(
            PointStamped, '/ball_point', self.point_callback, 10)
        self.sub_image = self.create_subscription(
            Image, '/image_topic', self.image_callback, 10)
        self.pub_image = self.create_publisher(
            Image, '/trajectory_image', 10)

    def point_callback(self, msg: PointStamped):
        x, y = msg.point.x, msg.point.y
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        self.latest_header = msg.header

        # --- State Classification ---
        if len(self.kf_history) == 0:
            self.state = STATE_STANDBY
        else:
            last_x, last_y = self.kf_history[-1]
            d = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)
            if d > self.flying_dist_thresh:
                self.state = STATE_FLYING

        if self.state == STATE_FLYING:
            self.missing_count = 0  # Reset missing counter
        self.state_str = STATE_NAMES[self.state]

        # --- Kalman Update ---
        if not self.kalman.initialized:
            self.kalman.init_state(x, y)
        else:
            # Compute dt from previous timestamp if available
            if len(self.time_history) > 0:
                self.dt = t - self.time_history[-1]
                if self.dt < 1e-3: self.dt = 1.0
                self.kalman.F = np.array([
                    [1, 0, self.dt, 0],
                    [0, 1, 0, self.dt],
                    [0, 0, 1,  0],
                    [0, 0, 0,  1]
                ])
            self.kalman.update([x, y])
            self.kalman.predict()

        # Save the filtered KF output and timestamp
        fx, fy = self.kalman.x[0, 0], self.kalman.x[1, 0]
        self.kf_history.append((fx, fy))
        self.time_history.append(t)

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        # --- Draw measured and KF trajectory ---
        if len(self.kf_history) >= 2:
            # Draw Kalman-filtered path (yellow)
            for i in range(1, len(self.kf_history)):
                cv2.line(
                    frame,
                    tuple(map(int, self.kf_history[i-1])),
                    tuple(map(int, self.kf_history[i])),
                    (0, 255, 255), 2
                )

        # --- LSF (Least Squares Fitting) ---
        if self.state == STATE_FLYING and len(self.kf_history) > 5:
            t0 = self.time_history[0]
            t_vals = np.array(self.time_history) - t0
            kf_arr = np.array(self.kf_history)

            # Remove points where time repeats or where nan/inf present
            valid = np.isfinite(t_vals) & np.isfinite(kf_arr[:,0]) & np.isfinite(kf_arr[:,1])
            t_vals = t_vals[valid]
            kf_arr = kf_arr[valid]

            # Remove duplicate or near-duplicate t_vals
            if len(t_vals) < 2 or np.all(np.abs(np.diff(t_vals)) < 1e-6):
                # Not enough valid, unique points to fit
                text = f"State: {self.state_str} | Prediction: insufficient data"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            else:
                # Fit x(t) and y(t) as lines: x(t) = a0 + a1*t
                x_coeffs = np.polyfit(t_vals, kf_arr[:, 0], 1)
                y_coeffs = np.polyfit(t_vals, kf_arr[:, 1], 1)

                # Predict trajectory for next 1 second (adjust as needed)
                t_pred = np.linspace(t_vals[0], t_vals[-1] + 1.0, 40)
                x_pred = np.polyval(x_coeffs, t_pred)
                y_pred = np.polyval(y_coeffs, t_pred)
                # Visualize as blue line
                pred_pts = np.stack([x_pred, y_pred], axis=1).astype(np.int32)
                for i in range(1, len(pred_pts)):
                    pt1 = tuple(pred_pts[i-1])
                    pt2 = tuple(pred_pts[i])
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
                cv2.circle(frame, tuple(pred_pts[-1]), 9, (0, 255, 0), -1)
                text = f"State: {self.state_str} | Predict @ ({pred_pts[-1][0]}, {pred_pts[-1][1]})"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            # Annotate only state
            text = f"State: {self.state_str}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


        # Publish the annotated image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            if self.latest_header:
                debug_msg.header = self.latest_header
            self.pub_image.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
