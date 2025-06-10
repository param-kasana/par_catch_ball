import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort
import os

class BallDetectorONNX(Node):
    def __init__(self):
        super().__init__('ball_detector_onnx')

        # Load ONNX model
        model_path = os.path.join(os.path.dirname(__file__), 'best.onnx')
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(PointStamped, '/ball_yolo', 10)

        self.get_logger().info("✅ YOLO ONNX model loaded.")

    def preprocess(self, img):
        img_resized = cv2.resize(img, (640, 640))
        img = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        return img, img_resized.shape[1], img_resized.shape[0]

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        img_input, width, height = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: img_input})[0]

        for det in outputs:
            conf = det[4]
            if conf < 0.4:
                continue
            x1, y1, x2, y2 = map(int, det[:4])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            pt = PointStamped()
            pt.header = msg.header
            pt.point.x = float(cx)
            pt.point.y = float(cy)
            pt.point.z = 0.0
            self.pub.publish(pt)
            self.get_logger().info(f"📍 Detected at ({cx}, {cy}) conf={conf:.2f}")
            break

def main(args=None):
    rclpy.init(args=args)
    node = BallDetectorONNX()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
