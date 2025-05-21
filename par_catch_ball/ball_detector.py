# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import PointStamped
# from cv_bridge import CvBridge
# import cv2
# import numpy as np

# class BallDetector(Node):
#     def __init__(self):
#         super().__init__('ball_detector')
#         self.bridge = CvBridge()

#         self.subscription = self.create_subscription(
#             Image,
#             '/image_topic',  # Change to your camera topic if needed
#             self.image_callback,
#             10
#         )
#         self.publisher = self.create_publisher(PointStamped, '/ball_point', 10)

#     def image_callback(self, msg):
#         try:
#             frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#         except Exception as e:
#             self.get_logger().error(f'Failed to convert image: {e}')
#             return

#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#         # Define orange HSV range – adjust as needed for lighting
#         lower_orange = np.array([5, 100, 90])
#         upper_orange = np.array([20, 255, 240])
#         mask = cv2.inRange(hsv, lower_orange, upper_orange)

#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         if contours:
#             largest = max(contours, key=cv2.contourArea)
#             M = cv2.moments(largest)
#             if M['m00'] != 0:
#                 cx = int(M['m10'] / M['m00'])
#                 cy = int(M['m01'] / M['m00'])

#                 point_msg = PointStamped()
#                 point_msg.header = msg.header
#                 point_msg.point.x = float(cx)
#                 point_msg.point.y = float(cy)
#                 point_msg.point.z = 0.0  # Use depth image to get real Z if available

#                 self.publisher.publish(point_msg)
#                 # self.get_logger().info(f'Published point: ({cx}, {cy})')

# def main(args=None):
#     rclpy.init(args=args)
#     node = BallDetector()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()


#----------------------------------------------------------------
# Visualisation

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge

import cv2
import numpy as np


class OrangeBallDetector(Node):
    def __init__(self):
        super().__init__('orange_ball_detector')
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            '/image_topic',  # Replace with your actual image topic
            self.image_callback,
            10
        )

        self.point_pub = self.create_publisher(PointStamped, '/ball_point', 10)
        self.marker_pub = self.create_publisher(Marker, '/ball_marker', 10)
        self.debug_img_pub = self.create_publisher(Image, '/image_debug', 10)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Tune these HSV values to match your ball under current lighting
        lower_orange = np.array([5, 150, 150])
        upper_orange = np.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # --- Publish point ---
                point_msg = PointStamped()
                point_msg.header = msg.header
                point_msg.point.x = float(cx)
                point_msg.point.y = float(cy)
                point_msg.point.z = 0.0
                self.point_pub.publish(point_msg)

                # --- Publish marker for RViz ---
                marker = Marker()
                marker.header = msg.header
                marker.ns = 'ball'
                marker.id = 0
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(cx)
                marker.pose.position.y = float(cy)
                marker.pose.position.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = 10.0
                marker.scale.y = 10.0
                marker.scale.z = 0.1
                marker.color.r = 1.0
                marker.color.g = 0.5
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker.lifetime.sec = 0
                self.marker_pub.publish(marker)

                # --- Draw detection on frame ---
                cv2.circle(frame, (cx, cy), 8, (0, 255, 0), 2)
                cv2.putText(frame, "Ball", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Publish debug image for RViz
        try:
            debug_img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            debug_img_msg.header = msg.header
            self.debug_img_pub.publish(debug_img_msg)
        except Exception as e:
            self.get_logger().error(f"Debug image conversion failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = OrangeBallDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
