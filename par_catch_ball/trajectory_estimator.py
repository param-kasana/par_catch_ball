#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker
import numpy as np
from collections import deque

class TrajectoryEstimator(Node):
    def __init__(self):
        super().__init__('trajectory_estimator')

        # Subscribe to ball position
        self.sub = self.create_subscription(
            PointStamped,
            '/ball_in_base',
            self.point_callback,
            10
        )

        # Publisher for trajectory visualization
        self.traj_pub = self.create_publisher(
            Marker,
            '/ball_predicted_trajectory',
            10
        )

        # Store recent points for fitting (last N points)
        self.history_len = 20
        self.points = deque(maxlen=self.history_len)
        self.times = deque(maxlen=self.history_len)
        self.start_time = None

        self.g = 9.81  # gravity

        self.timer = self.create_timer(0.1, self.publish_trajectory)

    def point_callback(self, msg):
        # Use msg.header.stamp as timestamp
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.start_time is None:
            self.start_time = t
        # Time relative to first point
        t_rel = t - self.start_time
        pt = [msg.point.x, msg.point.y, msg.point.z]
        self.points.append(pt)
        self.times.append(t_rel)

    def fit_trajectory(self):
        if len(self.points) < 5:
            return None  # Not enough points

        T = np.array(self.times)
        X = np.array([p[0] for p in self.points])
        Y = np.array([p[1] for p in self.points])
        Z = np.array([p[2] for p in self.points])

        # Linear fit for x, y
        A_lin = np.vstack([np.ones_like(T), T]).T
        x0, vx0 = np.linalg.lstsq(A_lin, X, rcond=None)[0]
        y0, vy0 = np.linalg.lstsq(A_lin, Y, rcond=None)[0]

        # Physics-constrained fit for z (gravity enforced)
        Z_gravity = Z + 0.5 * self.g * T**2
        z0, vz0 = np.linalg.lstsq(A_lin, Z_gravity, rcond=None)[0]

        return x0, vx0, y0, vy0, z0, vz0

    def predict_points(self, x0, vx0, y0, vy0, z0, vz0, t0, tf, dt=0.02):
        T = np.arange(t0, tf, dt)
        X = x0 + vx0 * T
        Y = y0 + vy0 * T
        Z = z0 + vz0 * T - 0.5 * self.g * T**2
        return X, Y, Z

    def publish_trajectory(self):
        fit = self.fit_trajectory()
        if fit is None:
            return  # Not enough points

        x0, vx0, y0, vy0, z0, vz0 = fit
        t0 = self.times[0]
        tf = self.times[-1] + 0.7  # Predict a bit into the future

        X, Y, Z = self.predict_points(x0, vx0, y0, vy0, z0, vz0, t0, tf)

        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "ball_trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.01  # Line width
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        marker.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in zip(X, Y, Z)]
        self.traj_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
