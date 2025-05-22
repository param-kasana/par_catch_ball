#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from collections import deque
import numpy as np
import cv2
import time

class TrajectoryEstimator3D(Node):
    def __init__(self):
        super().__init__('trajectory_estimator_3d')
        self.bridge = CvBridge()
        self.points = deque(maxlen=30)  # Store (timestamp, x, y, z)

        self.fx = self.fy = self.cx = self.cy = None
        self.camera_info_received = False
        self.latest_image = None

        # Subscribers
        self.sub_point = self.create_subscription(
            PointStamped,
            '/ball_point',
            self.point_callback,
            10
        )
        self.sub_image = self.create_subscription(
            Image,
            '/image_topic',
            self.image_callback,
            10
        )
        self.sub_cam_info = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )

        # Publisher
        self.pub_image = self.create_publisher(
            Image,
            '/trajectory_image',
            10
        )

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.camera_info_received = True
        self.sub_cam_info.destroy()  # We only need this once
        self.get_logger().info("✅ Camera intrinsics received.")

    def point_callback(self, msg: PointStamped):
        if not self.camera_info_received:
            return
        t = time.time()
        x, y, z = msg.point.x, msg.point.y, msg.point.z
        self.points.append((t, x, y, z))

    def image_callback(self, msg: Image):
        if not self.camera_info_received or len(self.points) < 5:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = image.copy()
            self.visualize_trajectory()
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def visualize_trajectory(self):
        image = self.latest_image.copy()
        data = np.array(self.points)
        t = data[:, 0] - data[0, 0]
        x = data[:, 1]
        y = data[:, 2]
        z = data[:, 3]

        # Fit using least squares
        A_lin = np.vstack([t, np.ones(len(t))]).T
        vx, cx = np.linalg.lstsq(A_lin, x, rcond=None)[0]
        vy, cy = np.linalg.lstsq(A_lin, y, rcond=None)[0]

        A_quad = np.vstack([t**2, t, np.ones(len(t))]).T
        az, bz, cz = np.linalg.lstsq(A_quad, z, rcond=None)[0]

        # Predict future 3D positions
        t_future = np.linspace(t[-1], t[-1] + 1.0, 30)
        x_future = vx * t_future + cx
        y_future = vy * t_future + cy
        z_future = az * t_future**2 + bz * t_future + cz

        # Project to image plane using camera intrinsics
        for i in range(1, len(t_future)):
            if z_future[i] <= 0:
                continue  # skip invalid depth

            u1 = int((x_future[i - 1] * self.fx) / z_future[i - 1] + self.cx)
            v1 = int((y_future[i - 1] * self.fy) / z_future[i - 1] + self.cy)
            u2 = int((x_future[i] * self.fx) / z_future[i] + self.cx)
            v2 = int((y_future[i] * self.fy) / z_future[i] + self.cy)

            if all(0 <= val < dim for val, dim in zip((u1, v1, u2, v2), (image.shape[1], image.shape[0], image.shape[1], image.shape[0]))):
                cv2.line(image, (u1, v1), (u2, v2), (255, 0, 0), 2)

        # Final predicted point
        final_u = int((x_future[-1] * self.fx) / z_future[-1] + self.cx)
        final_v = int((y_future[-1] * self.fy) / z_future[-1] + self.cy)
        if 0 <= final_u < image.shape[1] and 0 <= final_v < image.shape[0]:
            cv2.circle(image, (final_u, final_v), 6, (0, 0, 255), -1)

        # Publish the final image
        img_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        self.pub_image.publish(img_msg)
        self.get_logger().info("📤 Published predicted 3D trajectory.")

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryEstimator3D()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
