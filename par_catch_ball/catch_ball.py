#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, PointStamped
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import DisplayTrajectory, RobotState
from builtin_interfaces.msg import Duration
import math

class ArmController(Node):
    CONST_X = -0.4
    CONST_ORI_X = -0.49717557811531166
    CONST_ORI_Y = -0.5020731422162977
    CONST_ORI_Z = 0.50286146632379
    CONST_ORI_W = 0.49786479095980135
    REPLAN_THRESHOLD = 0.01  # [m] Minimal y/z change to trigger a replan

    def __init__(self):
        super().__init__('arm_controller')
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.traj_pub = self.create_publisher(DisplayTrajectory, '/display_planned_path', 10)
        self.current_goal = None  # Store last (y, z)
        self.current_goal_handle = None  # For canceling the current action

        self.get_logger().info("Waiting for MoveIt services...")
        while not self.cartesian_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /compute_cartesian_path...')
        while not self.execute_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for /execute_trajectory...')

        self.sub = self.create_subscription(PointStamped, '/ball_catch_point', self.catch_point_callback, 1)
        self.get_logger().info("Subscribed to /ball_catch_point.")

    def catch_point_callback(self, msg: PointStamped):
        # Only replan if new point is different enough
        y, z = msg.point.y, msg.point.z
        if self.current_goal is not None:
            last_y, last_z = self.current_goal
            dist = math.sqrt((y - last_y)**2 + (z - last_z)**2)
            if dist < self.REPLAN_THRESHOLD:
                self.get_logger().info(f"New catch point is very close to previous. Skipping replan.")
                return
        self.current_goal = (y, z)
        self.get_logger().info(f"New catch point received: y={y:.3f}, z={z:.3f}")

        # Cancel current trajectory if running
        if self.current_goal_handle is not None:
            self.get_logger().info("Cancelling current trajectory...")
            future = self.current_goal_handle.cancel_goal_async()
            future.add_done_callback(lambda fut: self.plan_and_execute(y, z))
            self.current_goal_handle = None
        else:
            self.plan_and_execute(y, z)

    def plan_and_execute(self, y, z):
        target_pose = Pose()
        target_pose.position.x = self.CONST_X
        target_pose.position.y = y
        target_pose.position.z = z
        target_pose.orientation.x = self.CONST_ORI_X
        target_pose.orientation.y = self.CONST_ORI_Y
        target_pose.orientation.z = self.CONST_ORI_Z
        target_pose.orientation.w = self.CONST_ORI_W
        self.move_to_pose(target_pose)

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

        request = GetCartesianPath.Request()
        request.group_name = 'ur_manipulator_end_effector'
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

        self.get_logger().info(f"Planning Cartesian path to y={pose.position.y:.3f}, z={pose.position.z:.3f}")
        future = self.cartesian_client.call_async(request)
        future.add_done_callback(self.handle_cartesian_path_response)

    def handle_cartesian_path_response(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Exception in cartesian path service: {e}")
            return

        if not response or len(response.solution.joint_trajectory.points) == 0:
            self.get_logger().error("Cartesian path planning failed.")
            return

        self.get_logger().info(f"Planned path fraction: {response.fraction:.2f}")

        # Optionally scale trajectory timing (speed control)
        scale = 0.8  # 80% speed
        for point in response.solution.joint_trajectory.points:
            original_sec = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            scaled_sec = original_sec / scale
            point.time_from_start = Duration()
            point.time_from_start.sec = int(scaled_sec)
            point.time_from_start.nanosec = int((scaled_sec - int(scaled_sec)) * 1e9)
            point.velocities = [v * scale for v in point.velocities]
            if point.accelerations:
                point.accelerations = [a * scale**2 for a in point.accelerations]

        traj_msg = DisplayTrajectory()
        traj_msg.trajectory_start = response.start_state
        traj_msg.trajectory.append(response.solution)
        self.traj_pub.publish(traj_msg)
        self.get_logger().info("Published trajectory to RViz")

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
            return

        if not goal_handle.accepted:
            self.get_logger().error("Execution goal rejected")
            return

        self.current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.handle_execution_result)

    def handle_execution_result(self, future):
        try:
            result = future.result()
            self.get_logger().info("Execution complete.")
        except Exception as e:
            self.get_logger().error(f"Exception in execution result: {e}")
        self.current_goal_handle = None

def main(args=None):
    rclpy.init(args=args)
    node = ArmController()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
