#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')
        self.publisher_ = self.create_publisher(Image, 'image_topic', 10)
        self.bridge = CvBridge()
        video_path = 'src/objects/ball_circle.mp4'  # Replace with your video path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video file at {video_path}")
            return
        self.timer = self.create_timer(1.0 / 60.0, self.timer_callback)  # Adjust to match your video's frame rate

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            # self.get_logger().info("End of video file reached.")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    video_publisher = VideoPublisher()
    rclpy.spin(video_publisher)
    video_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
