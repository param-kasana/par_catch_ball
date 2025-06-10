#!/usr/bin/env python3

import os
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np

class BallDetectorNode(Node):
    def __init__(self):
        super().__init__('ball_detector_node')

        # === YOLOv5 paths ===
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        yolov5_path = os.path.join(pkg_dir, 'yolov5n')
        weights_path = os.path.join(yolov5_path, 'weights', 'best.pt')

        sys.path.append(yolov5_path)
        self.get_logger().info(f"Loading YOLOv5 model from: {weights_path}")

        # === Load YOLOv5 model ===
        self.model = torch.hub.load(yolov5_path, 'custom', path=weights_path, source='local')
        self.model.conf = 0.4
        self.model.iou = 0.5

        self.bridge = CvBridge()

        # === ROS Interfaces ===
        self.sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(PointStamped, '/ball_yolo', 10)

        self.get_logger().info("✅ Ball Detector Node initialized.")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        results = self.model(frame)
        detections = results.xyxy[0].cpu().numpy()

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            pt = PointStamped()
            pt.header = msg.header
            pt.point.x = float(cx)
            pt.point.y = float(cy)
            pt.point.z = 0.0

            self.pub.publish(pt)
            self.get_logger().info(f"📍 Ball at ({cx}, {cy}) conf={conf:.2f}")
            break  # only first detection

def main(args=None):
    rclpy.init(args=args)
    node = BallDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
