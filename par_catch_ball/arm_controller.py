#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import DisplayTrajectory, RobotState
import json
import math
from builtin_interfaces.msg import Duration


class ExecuteCartesianPath(Node):
    def __init__(self):
        super().__init__('execute_cartesian_path_node')

        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.traj_pub = self.create_publisher(DisplayTrajectory, '/display_planned_path', 10)

        self.get_logger().info("Waiting for MoveIt services...")
        while not self.cartesian_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /compute_cartesian_path...')

        while not self.execute_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for /execute_trajectory...')

        self.send_cartesian_path_request()

    def send_cartesian_path_request(self):
        # Load pose from JSON
        try:
            with open('src/par_catch_ball/ee_pose.json', 'r') as json_file:
                pose_data = json.load(json_file)
        except Exception as e:
            self.get_logger().error(f"Failed to load pose: {e}")
            return

        # Normalize quaternion
        qx, qy, qz, qw = (
            pose_data["orientation"]["x"],
            pose_data["orientation"]["y"],
            pose_data["orientation"]["z"],
            pose_data["orientation"]["w"]
        )
        norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm

        # Build target pose
        target_pose = Pose()
        target_pose.position.x = pose_data["position"]["x"]
        target_pose.position.y = pose_data["position"]["y"]
        target_pose.position.z = pose_data["position"]["z"]
        target_pose.orientation.x = qx
        target_pose.orientation.y = qy
        target_pose.orientation.z = qz
        target_pose.orientation.w = qw

        # Build Cartesian path request
        request = GetCartesianPath.Request()
        request.group_name = 'ur_manipulator'
        request.link_name = 'tool0'
        request.header.frame_id = 'base_link'
        request.max_step = 0.01
        request.jump_threshold = 0.0
        request.avoid_collisions = False
        request.start_state = RobotState()
        request.start_state.is_diff = True
        request.waypoints.append(target_pose)

        self.get_logger().info("Planning Cartesian path...")
        future = self.cartesian_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        if not response or len(response.solution.joint_trajectory.points) == 0:
            self.get_logger().error("Cartesian path planning failed.")
            return

        self.get_logger().info(f"Planned path fraction: {response.fraction:.2f}")

        # Optionally scale trajectory timing (speed control)
        scale = 0.5  # 50% speed
        for point in response.solution.joint_trajectory.points:
            original_sec = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            scaled_sec = original_sec / scale
            point.time_from_start = Duration()
            point.time_from_start.sec = int(scaled_sec)
            point.time_from_start.nanosec = int((scaled_sec - int(scaled_sec)) * 1e9)
            point.velocities = [v * scale for v in point.velocities]
            if point.accelerations:
                point.accelerations = [a * scale**2 for a in point.accelerations]

        # Visualize in RViz
        traj_msg = DisplayTrajectory()
        traj_msg.trajectory_start = response.start_state
        traj_msg.trajectory.append(response.solution)
        self.traj_pub.publish(traj_msg)
        self.get_logger().info("Published trajectory to RViz")

        # Execute trajectory
        self.get_logger().info("Executing trajectory...")
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = response.solution
        exec_future = self.execute_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, exec_future)

        goal_handle = exec_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Execution goal rejected")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        self.get_logger().info("Execution complete.")


def main(args=None):
    rclpy.init(args=args)
    node = ExecuteCartesianPath()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
