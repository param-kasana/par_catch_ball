import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

class EEPositionPrinter(Node):
    def __init__(self):
        super().__init__('ee_position_printer')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0, self.print_ee_pose)

    def print_ee_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
            pos = trans.transform.translation
            self.get_logger().info(f"End effector at: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")

def main():
    rclpy.init()
    node = EEPositionPrinter()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
