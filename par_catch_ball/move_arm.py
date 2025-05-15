#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient

from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

class PoseGoalClient(Node):
    def __init__(self):
        super().__init__('pose_goal_client')
        self._client = ActionClient(self, MoveGroup, '/move_action')

        self.timer = self.create_timer(2.0, self.send_goal)  # Delay for MoveIt to be ready
        self.sent = False

    def send_goal(self):
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveGroup action server not available')
            return

        if self.sent:
            return

        request = MotionPlanRequest()
        request.group_name = 'ur_manipulator'

        constraint = Constraints()

        # Position constraint (bounding box around target point)
        pos_con = PositionConstraint()
        pos_con.header.frame_id = "base_link"
        pos_con.link_name = "tool0"

        shape = SolidPrimitive()
        shape.type = SolidPrimitive.BOX
        shape.dimensions = [0.001, 0.001, 0.001]

        pose = Pose()
        pose.position.x = 0.4
        pose.position.y = 0.0
        pose.position.z = 0.3
        pose.orientation.w = 1.0

        bv = BoundingVolume()
        bv.primitives.append(shape)
        bv.primitive_poses.append(pose)

        pos_con.constraint_region = bv
        pos_con.weight = 1.0

        constraint.position_constraints.append(pos_con)
        request.goal_constraints.append(constraint)
        request.allowed_planning_time = 5.0

        goal_msg = MoveGroup.Goal()
        goal_msg.request = request
        goal_msg.planning_options.plan_only = False
        goal_msg.planning_options.look_around = False
        goal_msg.planning_options.replan = False

        self._client.send_goal_async(goal_msg)
        self.get_logger().info("✅ Sent pose goal to MoveIt")
        self.sent = True


def main(args=None):
    rclpy.init(args=args)
    node = PoseGoalClient()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
