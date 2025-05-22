#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Point
from par_interfaces.action import WaypointMove
from par_interfaces.msg import WaypointPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
import math
import numpy as np
from collections import deque

class BallTracker(Node):
    def __init__(self):
        super().__init__('ball_tracker')

        # Parameters
        self.safe_z_offset = 0.3  # distance BELOW the ball (in meters)
        self.min_move_distance = 0.05  # meters
        self.min_time_interval = 0.5  # seconds
        self.smoothing_window = 5  # number of past positions to average

        # Tracking buffers
        self.recent_points = deque(maxlen=self.smoothing_window)
        self.last_target = None
        self.last_goal_time = self.get_clock().now()

        # Subscribers and clients
        self.subscription = self.create_subscription(
            PointStamped,
            '/ball_in_base',
            self.ball_callback,
            10
        )

        self.client = ActionClient(self, WaypointMove, '/par_moveit/waypoint_move')
        self.get_logger().info("Smoothed Ball Tracker started.")

    def ball_callback(self, msg: PointStamped):
        # Offset to move BELOW the ball
        p = msg.point
        p.z -= self.safe_z_offset
        if p.z < 0.05:
            p.z = 0.05  # safety clamp above table

        self.recent_points.append(p)

        if len(self.recent_points) < self.smoothing_window:
            return  # wait until we have enough points

        # Compute averaged position
        avg_point = Point()
        avg_point.x = np.mean([pt.x for pt in self.recent_points])
        avg_point.y = np.mean([pt.y for pt in self.recent_points])
        avg_point.z = np.mean([pt.z for pt in self.recent_points])

        now = self.get_clock().now()
        time_diff = (now - self.last_goal_time).nanoseconds / 1e9

        # Check distance moved
        if self.last_target is not None:
            dist = math.sqrt(
                (avg_point.x - self.last_target.x)**2 +
                (avg_point.y - self.last_target.y)**2 +
                (avg_point.z - self.last_target.z)**2
            )
            if dist < self.min_move_distance or time_diff < self.min_time_interval:
                return  # too soon or too small a change

        self.last_target = avg_point
        self.last_goal_time = now
        self.send_move_goal(avg_point)

    def send_move_goal(self, point):
        self.client.wait_for_server()

        goal = WaypointMove.Goal()
        goal.target_pose = WaypointPose()
        goal.target_pose.position = point
        goal.target_pose.rotation = 0.0  # keep gripper level (adjust if needed)

        self.get_logger().info(
            f"Moving to smoothed point: x={point.x:.2f}, y={point.y:.2f}, z={point.z:.2f}"
        )

        self.client.send_goal_async(goal)

def main(args=None):
    rclpy.init(args=args)
    node = BallTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
