#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
import json  # Add this import

class EEPositionReader(Node):
    def __init__(self):
        super().__init__('ee_pose_reader_world')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0, self.read_ee_pose)

    def read_ee_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
            pos = trans.transform.translation
            ori = trans.transform.rotation

            # Log the pose
            self.get_logger().info(
                f"\nEnd-effector pose in 'base_link' frame:\n"
                f"  Position:\n"
                f"    x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}\n"
                f"  Orientation (normalized quaternion):\n"
                f"    x={ori.x:.3f}, y={ori.y:.3f}, z={ori.z:.3f}, w={ori.w:.3f}"
            )

            # Save the pose to a JSON file
            pose_data = {
                "position": {"x": pos.x, "y": pos.y, "z": pos.z},
                "orientation": {"x": ori.x, "y": ori.y, "z": ori.z, "w": ori.w}
            }
            with open('src/par_catch_ball/ee_pose.json', 'w') as json_file:
                json.dump(pose_data, json_file, indent=4)

        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")

def main():
    rclpy.init()
    node = EEPositionReader()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
