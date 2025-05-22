#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from image_geometry import PinholeCameraModel

class BallDetector(Node):
    def __init__(self):
        super().__init__('ball_detector')

        # Set up topics
        self.rgb_topic = '/camera/camera/color/image_raw'
        self.depth_topic = '/camera/camera/depth/image_rect_raw'
        self.camera_info_topic = '/camera/camera/color/camera_info'
        self.output_topic = '/ball'
        self.debug_topic = '/ball_debug'

        # Camera model and bridge
        self.camera_model = PinholeCameraModel()
        self.bridge = CvBridge()
        self.camera_info_received = False

        # Publishers and subscribers
        self.ball_pub = self.create_publisher(PointStamped, self.output_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_topic, 10)

        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)

        self.latest_depth = None

        self.get_logger().info("3D Ball Detector started. Waiting for camera info...")

    def camera_info_callback(self, msg):
        if not self.camera_info_received:
            self.camera_model.fromCameraInfo(msg)
            self.camera_info_received = True
            self.get_logger().info("Camera info received.")

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def rgb_callback(self, msg):
        if not self.camera_info_received or self.latest_depth is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"RGB conversion failed: {e}")
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([5, 150, 150])
        upper_orange = np.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest)

            if radius > 5:
                u, v = int(x), int(y)

                # Depth in meters
                depth = self.latest_depth[v, u] / 1000.0
                if depth == 0.0 or np.isnan(depth) or depth > 5.0:
                    return

                # Project to 3D
                ray = self.camera_model.projectPixelTo3dRay((u, v))
                point_camera = np.array(ray) * depth

                # Publish
                pt = PointStamped()
                pt.header = msg.header
                pt.header.frame_id = 'camera_color_optical_frame'
                pt.point.x = float(point_camera[0])
                pt.point.y = float(point_camera[1])
                pt.point.z = float(point_camera[2])
                self.ball_pub.publish(pt)

                # Draw on debug image
                cv2.circle(frame, (u, v), int(radius), (0, 255, 0), 2)
                cv2.putText(frame, f"({pt.point.x:.2f}, {pt.point.y:.2f}, {pt.point.z:.2f})", (u + 10, v - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        debug_img = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        debug_img.header = msg.header
        self.debug_pub.publish(debug_img)

def main(args=None):
    rclpy.init(args=args)
    node = BallDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
