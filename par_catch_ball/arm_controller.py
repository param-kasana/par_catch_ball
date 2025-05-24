#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import DisplayTrajectory, RobotState
from builtin_interfaces.msg import Duration
import math

class BallFollower(Node):
    def __init__(self):
        super().__init__('ball_follower_node')
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.traj_pub = self.create_publisher(DisplayTrajectory, '/display_planned_path', 10)
        self.motion_in_progress = False

        self.get_logger().info("Waiting for MoveIt services...")
        while not self.cartesian_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /compute_cartesian_path...')
        while not self.execute_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for /execute_trajectory...')

        self.sub = self.create_subscription(
            Pose,
            '/ball_in_base',
            self.ball_pose_callback,
            1
        )
        self.get_logger().info("Subscribed to /ball_in_base.")

    def ball_pose_callback(self, msg):
        if self.motion_in_progress:
            self.get_logger().info("Motion in progress, skipping new pose.")
            return
        self.motion_in_progress = True
        self.get_logger().info(
            f"Received new ball pose: x={msg.position.x:.3f}, y={msg.position.y:.3f}, z={msg.position.z:.3f}"
        )
        self.move_to_pose(msg)

    def move_to_pose(self, target_pose):
        # Normalize quaternion
        qx, qy, qz, qw = (
            target_pose.orientation.x,
            target_pose.orientation.y,
            target_pose.orientation.z,
            target_pose.orientation.w
        )
        norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm

        # Prepare MoveIt Cartesian path request
        request = GetCartesianPath.Request()
        request.group_name = 'ur_manipulator'
        request.link_name = 'tool0'
        request.header.frame_id = 'base_link'
        request.max_step = 0.01
        request.jump_threshold = 0.0
        request.avoid_collisions = False
        request.start_state = RobotState()
        request.start_state.is_diff = True
        pose = Pose()
        pose.position = target_pose.position
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        request.waypoints.append(pose)

        self.get_logger().info("Planning Cartesian path...")
        future = self.cartesian_client.call_async(request)
        future.add_done_callback(self.handle_cartesian_path_response)

    def handle_cartesian_path_response(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Exception in cartesian path service: {e}")
            self.motion_in_progress = False
            return

        if not response or len(response.solution.joint_trajectory.points) == 0:
            self.get_logger().error("Cartesian path planning failed.")
            self.motion_in_progress = False
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
        exec_future.add_done_callback(self.handle_execute_trajectory_response)

    def handle_execute_trajectory_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f"Exception in execute trajectory action: {e}")
            self.motion_in_progress = False
            return

        if not goal_handle.accepted:
            self.get_logger().error("Execution goal rejected")
            self.motion_in_progress = False
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.handle_execution_result)

    def handle_execution_result(self, future):
        try:
            result = future.result()
            self.get_logger().info("Execution complete.")
        except Exception as e:
            self.get_logger().error(f"Exception in execution result: {e}")
        self.motion_in_progress = False

def main(args=None):
    rclpy.init(args=args)
    node = BallFollower()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
