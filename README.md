# ROS2 Node Structure
`ball_detector.py`: detects and tracks ball in camera frame.

`trajectory_estimator.py`: fits model, publishes predicted pose and time.

`interception_planner.py`: computes reachable poses, calls MoveIt.

`arm_controller.py`: handles trajectory execution and feedback.

