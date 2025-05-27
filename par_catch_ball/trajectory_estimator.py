#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker
import numpy as np
from collections import deque
import csv
import os
from datetime import datetime

class TrajectoryEstimator(Node):
    def __init__(self):
        super().__init__('trajectory_estimator')

        # Subscribers and publishers
        self.sub = self.create_subscription(PointStamped, '/ball_in_base', self.point_callback, 10)
        self.traj_pub = self.create_publisher(Marker, '/ball_predicted_trajectory', 10)
        self.ball_points_pub = self.create_publisher(Marker, '/ball_observed_points', 10)
        self.pred_points_pub = self.create_publisher(Marker, '/ball_predicted_points', 10)

        # Data for trajectory fitting
        self.history_len = 5
        self.points = deque(maxlen=self.history_len)
        self.times = deque(maxlen=self.history_len)
        self.start_time = None
        self.g = 9.81

        # --- File/log setup with timestamp ---
        script_dir = os.path.dirname(os.path.realpath(__file__))
        log_dir = os.path.join(script_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.observed_file = os.path.join(log_dir, f'observed_points_{timestamp}.csv')
        self.predicted_file = os.path.join(log_dir, f'predicted_trajectory_{timestamp}.csv')
        self.catch_points_file = os.path.join(log_dir, f'catch_points_{timestamp}.csv')
        # Write headers
        with open(self.observed_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['t', 'x', 'y', 'z'])
        with open(self.predicted_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['t', 'x', 'y', 'z'])
        with open(self.catch_points_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['t', 'x', 'y', 'z'])

        self.timer = self.create_timer(0.1, self.publish_markers)

    def point_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.start_time is None:
            self.start_time = t
        t_rel = t - self.start_time
        pt = [msg.point.x, msg.point.y, msg.point.z]
        self.points.append(pt)
        self.times.append(t_rel)

        os.makedirs(os.path.dirname(self.observed_file), exist_ok=True)
        # Save to observed points file
        with open(self.observed_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([t_rel, pt[0], pt[1], pt[2]])

    def fit_trajectory(self):
        if len(self.points) < 5:
            return None

        T = np.array(self.times)
        X = np.array([p[0] for p in self.points])
        Y = np.array([p[1] for p in self.points])
        Z = np.array([p[2] for p in self.points])

        A_lin = np.vstack([np.ones_like(T), T]).T
        x0, vx0 = np.linalg.lstsq(A_lin, X, rcond=None)[0]
        y0, vy0 = np.linalg.lstsq(A_lin, Y, rcond=None)[0]

        Z_gravity = Z + 0.5 * self.g * T**2
        z0, vz0 = np.linalg.lstsq(A_lin, Z_gravity, rcond=None)[0]
        return x0, vx0, y0, vy0, z0, vz0

    def predict_points(self, x0, vx0, y0, vy0, z0, vz0, t0, tf, dt=0.02):
        T = np.arange(t0, tf, dt)
        X = x0 + vx0 * T
        Y = y0 + vy0 * T
        Z = z0 + vz0 * T - 0.5 * self.g * T**2
        return T, X, Y, Z

    def publish_markers(self):
        # Publish observed points marker (all observed, no filter)
        ball_marker = Marker()
        ball_marker.header.frame_id = "base_link"
        ball_marker.header.stamp = self.get_clock().now().to_msg()
        ball_marker.ns = "ball_points"
        ball_marker.id = 1
        ball_marker.type = Marker.POINTS
        ball_marker.action = Marker.ADD
        ball_marker.scale.x = 0.03
        ball_marker.scale.y = 0.03
        ball_marker.color.a = 1.0
        ball_marker.color.r = 1.0  # orange
        ball_marker.color.g = 0.55
        ball_marker.color.b = 0.0
        ball_marker.points = [Point(x=p[0], y=p[1], z=p[2]) for p in self.points]
        self.ball_points_pub.publish(ball_marker)

        fit = self.fit_trajectory()
        if fit is None:
            return

        x0, vx0, y0, vy0, z0, vz0 = fit
        t0 = self.times[0]
        tf = self.times[-1] + 0.7

        # Predict the full trajectory
        T, X, Y, Z = self.predict_points(x0, vx0, y0, vy0, z0, vz0, t0, tf)

        # Save full predicted trajectory to file
        with open(self.predicted_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['t', 'x', 'y', 'z'])
            for t, x, y, z in zip(T, X, Y, Z):
                writer.writerow([t, x, y, z])

        # Publish full predicted trajectory as LINE_STRIP
        traj_marker = Marker()
        traj_marker.header.frame_id = "base_link"
        traj_marker.header.stamp = self.get_clock().now().to_msg()
        traj_marker.ns = "ball_trajectory"
        traj_marker.id = 0
        traj_marker.type = Marker.LINE_STRIP
        traj_marker.action = Marker.ADD
        traj_marker.scale.x = 0.01
        traj_marker.color.a = 1.0
        traj_marker.color.r = 0.0
        traj_marker.color.g = 1.0
        traj_marker.color.b = 0.0
        traj_marker.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in zip(X, Y, Z)]
        self.traj_pub.publish(traj_marker)

        # --- Catch points: 0.2 <= z <= 0.4, x < -1.0 ---
        catch_mask = (Z >= 0.2) & (Z <= 0.4) & (X < -0.7)
        X_catch = X[catch_mask]
        Y_catch = Y[catch_mask]
        Z_catch = Z[catch_mask]
        T_catch = T[catch_mask]

        # Save catch points to file
        with open(self.catch_points_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['t', 'x', 'y', 'z'])
            for t, x, y, z in zip(T_catch, X_catch, Y_catch, Z_catch):
                writer.writerow([t, x, y, z])

        # Publish predicted points marker (catch points only)
        pred_marker = Marker()
        pred_marker.header.frame_id = "base_link"
        pred_marker.header.stamp = self.get_clock().now().to_msg()
        pred_marker.ns = "predicted_points"
        pred_marker.id = 2
        pred_marker.type = Marker.POINTS
        pred_marker.action = Marker.ADD
        pred_marker.scale.x = 0.03
        pred_marker.scale.y = 0.03
        pred_marker.color.a = 1.0
        pred_marker.color.r = 0.0
        pred_marker.color.g = 1.0
        pred_marker.color.b = 0.0
        pred_marker.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in zip(X_catch, Y_catch, Z_catch)]
        self.pred_points_pub.publish(pred_marker)

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
