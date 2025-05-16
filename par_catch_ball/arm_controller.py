#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    BoundingVolume,
    MoveItErrorCodes,
)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, Point, Quaternion

class FastInterceptMover(Node):
    def __init__(self, x: float, y: float, z: float):
        super().__init__('fast_intercept_mover')
        self._client = ActionClient(self, MoveGroup, '/move_action')
        self._point = (x, y, z)

        # Will hold our original MotionPlanRequest so we can reuse it for execution
        self._request = None
        # Timer to kick off send_goal once; we'll cancel it immediately in _send_goal
        self._timer = self.create_timer(1.0, self._send_goal)
        self._sent = False

    def _send_goal(self):
        # Ensure we only send once
        if self._sent:
            return
        self._sent = True
        self._timer.cancel()

        # Wait for MoveIt action server
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("❌ MoveGroup action server not available.")
            rclpy.shutdown()
            return

        # Build the planning request
        req = MotionPlanRequest()
        req.group_name = 'ur_manipulator'
        req.allowed_planning_time = 3.0

        # Max‐out speed and acceleration
        req.max_velocity_scaling_factor = 1.0
        req.max_acceleration_scaling_factor = 1.0

        # Create a tiny box constraint around the target point
        pc = PositionConstraint()
        pc.header.frame_id = 'base_link'
        pc.link_name = 'tool0'

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.002, 0.002, 0.002]  # 2 mm cube

        pose = Pose()
        pose.position = Point(x=self._point[0], y=self._point[1], z=self._point[2])
        pose.orientation = Quaternion(w=1.0)  # keep net level

        bv = BoundingVolume()
        bv.primitives.append(box)
        bv.primitive_poses.append(pose)
        pc.constraint_region = bv
        pc.weight = 1.0

        cons = Constraints()
        cons.position_constraints.append(pc)
        req.goal_constraints.append(cons)

        # Save request for reuse
        self._request = req

        # First pass: plan only
        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only  = True
        goal.planning_options.look_around = False
        goal.planning_options.replan     = False

        self.get_logger().info(f"📝 Planning trajectory to {self._point} ...")
        plan_future = self._client.send_goal_async(goal)
        plan_future.add_done_callback(self._on_plan_response)

    def _on_plan_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("❌ Planning request was rejected.")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_plan_result)

    def _on_plan_result(self, future):
        result = future.result().result
        code   = result.error_code.val

        if code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"❌ Planning failed (code {code}). Goal not feasible.")
            rclpy.shutdown()
            return

        self.get_logger().info("✅ Planning successful. Executing motion...")

        # Second pass: execute using same request
        exec_goal = MoveGroup.Goal()
        exec_goal.request = self._request
        exec_goal.planning_options.plan_only  = False
        exec_goal.planning_options.look_around = False
        exec_goal.planning_options.replan     = False

        exec_future = self._client.send_goal_async(exec_goal)
        exec_future.add_done_callback(self._on_execute_response)

    def _on_execute_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("❌ Execution request was rejected.")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future):
        result = future.result().result
        code   = result.error_code.val

        if code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"❌ Motion execution failed (code {code}).")
        else:
            self.get_logger().info("✅ Motion completed successfully.")
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    # ← Replace with your computed intercept point
    node = FastInterceptMover(0.0, 0.5, 0.5)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
