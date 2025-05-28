#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PointStamped, Pose
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import DisplayTrajectory, RobotState
from builtin_interfaces.msg import Duration
import math
import time
from collections import deque

def clamp_to_circle(y, z, center_y, center_z, radius):
    dy = y - center_y
    dz = z - center_z
    dist = math.sqrt(dy**2 + dz**2)
    if dist > radius:
        scale = radius / dist
        y = center_y + dy * scale
        z = center_z + dz * scale
    return y, z

class BallFollower(Node):
    # ======= CONSTANTS FOR ARM POSITIONING ========
    CONST_X = -0.2
    CONST_ORI_X = -0.49717557811531166
    CONST_ORI_Y = -0.5020731422162977
    CONST_ORI_Z = 0.50286146632379
    CONST_ORI_W = 0.49786479095980135
    # ================================================
    AVG_WINDOW = 10
    MIN_DISPLACEMENT = 0.05  # 5 cm
    PROCESS_INTERVAL = 0.2   # seconds

    CENTER_Y = 0.14362462490424646
    CENTER_Z = 0.742090833804857
    RADIUS = 0.3

    def __init__(self):
        super().__init__('ball_follower_node')
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.traj_pub = self.create_publisher(DisplayTrajectory, '/display_planned_path', 10)
        self.motion_in_progress = False
        self.target_queue = deque()

        self.get_logger().info("Waiting for MoveIt services...")
        while not self.cartesian_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /compute_cartesian_path...')
        while not self.execute_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for /execute_trajectory...')

        self.sub = self.create_subscription(PointStamped, '/ball_in_base', self.ball_point_callback, 1)
        self.get_logger().info("Subscribed to /ball_in_base.")

        self.positions_y = deque(maxlen=self.AVG_WINDOW)
        self.positions_z = deque(maxlen=self.AVG_WINDOW)
        self.last_sent_y = None
        self.last_sent_z = None
        self.last_process_time = time.time()

    def ball_point_callback(self, msg: PointStamped):
        now = time.time()
        if now - self.last_process_time < self.PROCESS_INTERVAL:
            return
        self.last_process_time = now

        self.positions_y.append(msg.point.y)
        self.positions_z.append(msg.point.z)

        if len(self.positions_y) < self.AVG_WINDOW:
            self.get_logger().info(f"Buffering positions ({len(self.positions_y)}/{self.AVG_WINDOW})")
            return

        avg_y = sum(self.positions_y) / self.AVG_WINDOW
        avg_z = sum(self.positions_z) / self.AVG_WINDOW

        clamped_y, clamped_z = clamp_to_circle(avg_y, avg_z, self.CENTER_Y, self.CENTER_Z, self.RADIUS)
        if (clamped_y, clamped_z) != (avg_y, avg_z):
            self.get_logger().warn(f"Target outside area; clamped to ({clamped_y:.3f}, {clamped_z:.3f})")

        if self.last_sent_y is not None and self.last_sent_z is not None:
            dist = math.sqrt((clamped_y - self.last_sent_y)**2 + (clamped_z - self.last_sent_z)**2)
            if dist < self.MIN_DISPLACEMENT:
                self.get_logger().info(f"Displacement {dist:.3f} below threshold; skipping.")
                return

        self.last_sent_y = clamped_y
        self.last_sent_z = clamped_z

        target_pose = Pose()
        target_pose.position.x = self.CONST_X
        target_pose.position.y = clamped_y
        target_pose.position.z = clamped_z
        target_pose.orientation.x = self.CONST_ORI_X
        target_pose.orientation.y = self.CONST_ORI_Y
        target_pose.orientation.z = self.CONST_ORI_Z
        target_pose.orientation.w = self.CONST_ORI_W

        self.target_queue.append((target_pose, msg.header.frame_id))
        self.get_logger().info(f"Queued target y={clamped_y:.3f}, z={clamped_z:.3f}")

        if not self.motion_in_progress:
            self.process_next_trajectory()

    def process_next_trajectory(self):
        if not self.target_queue:
            self.get_logger().info("Queue empty.")
            self.motion_in_progress = False
            return

        target_pose, frame_id = self.target_queue.popleft()
        self.motion_in_progress = True
        self.move_to_pose(target_pose, frame_id)

    def move_to_pose(self, target_pose, frame_id):
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
        request.header.frame_id = frame_id
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

        self.get_logger().info(f"Planning path in frame: {frame_id}")
        future = self.cartesian_client.call_async(request)
        future.add_done_callback(self.handle_cartesian_path_response)

    def handle_cartesian_path_response(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Path planning failed: {e}")
            self.motion_in_progress = False
            return

        if not response or len(response.solution.joint_trajectory.points) == 0:
            self.get_logger().error("Empty Cartesian path.")
            self.motion_in_progress = False
            return

        self.get_logger().info(f"Planned path fraction: {response.fraction:.2f}")

        scale = 0.7
        for point in response.solution.joint_trajectory.points:
            original_sec = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            scaled_sec = original_sec / scale
            point.time_from_start.sec = int(scaled_sec)
            point.time_from_start.nanosec = int((scaled_sec - int(scaled_sec)) * 1e9)
            point.velocities = [v * scale for v in point.velocities]
            if point.accelerations:
                point.accelerations = [a * scale**2 for a in point.accelerations]

        traj_msg = DisplayTrajectory()
        traj_msg.trajectory_start = response.start_state
        traj_msg.trajectory.append(response.solution)
        self.traj_pub.publish(traj_msg)
        self.get_logger().info("Published trajectory to RViz.")

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = response.solution
        exec_future = self.execute_client.send_goal_async(goal)
        exec_future.add_done_callback(self.handle_execute_trajectory_response)

    def handle_execute_trajectory_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f"Execution goal error: {e}")
            self.motion_in_progress = False
            return

        if not goal_handle.accepted:
            self.get_logger().error("Execution goal rejected.")
            self.motion_in_progress = False
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.handle_execution_result)

    def handle_execution_result(self, future):
        try:
            result = future.result()
            self.get_logger().info("Execution complete.")
        except Exception as e:
            self.get_logger().error(f"Execution result error: {e}")
        self.process_next_trajectory()

def main(args=None):
    rclpy.init(args=args)
    node = BallFollower()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
