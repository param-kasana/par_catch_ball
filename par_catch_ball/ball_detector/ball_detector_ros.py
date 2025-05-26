#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from image_geometry import PinholeCameraModel
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from rclpy.duration import Duration

class BallDetector(Node):
    def __init__(self):
        super().__init__('ball_detector_hough')

        self.rgb_topic = '/camera/camera/color/image_raw'
        self.depth_topic = '/camera/camera/depth/image_rect_raw'
        self.camera_info_topic = '/camera/camera/color/camera_info'
        self.output_topic = '/ball'
        self.debug_topic = '/ball_debug'

        self.camera_model = PinholeCameraModel()
        self.bridge = CvBridge()
        self.camera_info_received = False
        self.latest_depth = None

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.ball_pub = self.create_publisher(PointStamped, self.output_topic, 10)
        self.base_pub = self.create_publisher(PointStamped, '/ball_in_base', 10)
        self.debug_pub = self.create_publisher(Image, self.debug_topic, 10)

        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)

        self.get_logger().info(" Ball detector with Hough Circle Transform started.")

    def camera_info_callback(self, msg):
        if not self.camera_info_received:
            self.camera_model.fromCameraInfo(msg)
            self.camera_info_received = True
            self.get_logger().info(" Camera info received.")

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f" Depth conversion failed: {e}")

    def rgb_callback(self, msg):
        if not self.camera_info_received or self.latest_depth is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f" RGB conversion failed: {e}")
            return

        # Step 1: HSV Filtering
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([2, 150, 180])
        upper_orange = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Step 2: Blur and detect circles
        blurred = cv2.GaussianBlur(mask, (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                   param1=100, param2=15, minRadius=10, maxRadius=60)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :1]:  # Only the first circle
                u, v = int(x), int(y)
                radius = int(r)

                # Depth at center
                depth = self.latest_depth[v, u] / 1000.0  # in meters
                if depth == 0.0 or np.isnan(depth) or depth > 5.0:
                    continue

                # Project to 3D
                ray = self.camera_model.projectPixelTo3dRay((u, v))
                point_camera = np.array(ray) * depth

                pt = PointStamped()
                pt.header = msg.header
                pt.header.frame_id = 'camera_color_optical_frame'
                pt.point.x = float(point_camera[0])
                pt.point.y = float(point_camera[1])
                pt.point.z = float(point_camera[2])

                self.ball_pub.publish(pt)

                try:
                    transform = self.tf_buffer.lookup_transform(
                        'base_link', pt.header.frame_id, rclpy.time.Time())
                    pt_base = do_transform_point(pt, transform)
                    self.base_pub.publish(pt_base)
                except Exception as e:
                    self.get_logger().warn(f" TF transform failed: {e}")

                # Draw circle and label
                cv2.circle(frame, (u, v), radius, (0, 255, 0), 2)
                cv2.putText(frame, f"Ball @ ({u}, {v})", (u + 10, v - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                break

        # Publish debug image
        try:
            debug_img = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            debug_img.header = msg.header
            self.debug_pub.publish(debug_img)
        except Exception as e:
            self.get_logger().warn(f" Debug image conversion failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = BallDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
