#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from collections import deque
import numpy as np
import time

class TrajectoryEstimator(Node):
    def __init__(self):
        super().__init__('trajectory_estimator')

        # Parameters
        self.declare_parameter('ball_topic', '/ball')
        self.declare_parameter('catch_point_topic', '/catch_point')
        self.declare_parameter('catch_plane_z', 0.5)  # meters
        self.declare_parameter('max_points', 20)

        self.catch_plane_z = self.get_parameter('catch_plane_z').value
        self.max_points = self.get_parameter('max_points').value

        self.points = deque(maxlen=self.max_points)
        self.times = deque(maxlen=self.max_points)

        self.ball_sub = self.create_subscription(PointStamped, self.get_parameter('ball_topic').value, self.ball_callback, 10)
        self.catch_pub = self.create_publisher(PointStamped, self.get_parameter('catch_point_topic').value, 10)

        self.get_logger().info("Trajectory estimator node started.")

    def ball_callback(self, msg: PointStamped):
        # Store ball position and timestamp (in seconds)
        now = self.get_clock().now().seconds_nanoseconds()[0] + \
              self.get_clock().now().seconds_nanoseconds()[1] * 1e-9
        self.points.append((msg.point.x, msg.point.y, msg.point.z))
        self.times.append(now)

        # Fit and predict when we have enough points
        if len(self.points) >= 5:
            self.predict_intercept(msg.header)

    def predict_intercept(self, header):
        # Prepare data
        times = np.array(self.times)
        times -= times[0]  # make relative
        points = np.array(self.points)

        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]

        # Fit x(t), y(t), z(t) with quadratic or linear
        try:
            coeffs_x = np.polyfit(times, xs, 2)
            coeffs_y = np.polyfit(times, ys, 2)
            coeffs_z = np.polyfit(times, zs, 2)
        except Exception as e:
            self.get_logger().warn(f"Fit error: {e}")
            return

        # Solve for time when z(t) = catch_plane_z
        a, b, c = coeffs_z
        c -= self.catch_plane_z
        roots = np.roots([a, b, c])

        # Find the earliest valid future root
        t_catch = None
        for r in roots:
            if np.isreal(r) and r > 0:
                t_catch = np.real(r)
                break

        if t_catch is None:
            self.get_logger().warn("No valid intercept time found.")
            return

        # Predict x, y at intercept time
        x_catch = np.polyval(coeffs_x, t_catch)
        y_catch = np.polyval(coeffs_y, t_catch)
        z_catch = self.catch_plane_z

        # Publish catch point
        catch_point = PointStamped()
        catch_point.header = header
        catch_point.point.x = float(x_catch)
        catch_point.point.y = float(y_catch)
        catch_point.point.z = float(z_catch)

        self.catch_pub.publish(catch_point)
        self.get_logger().info(f"Predicted catch at x={x_catch:.2f}, y={y_catch:.2f}, z={z_catch:.2f} in {t_catch:.2f}s")

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
