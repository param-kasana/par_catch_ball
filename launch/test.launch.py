from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os

def generate_launch_description():
    # Get the package share directory for rviz config
    rviz_config_path = os.path.join(
        os.getenv('AMENT_PREFIX_PATH', '').split(':')[0],  # first workspace in the path
        'share', 'par_catch_ball', 'rviz', 'par_catch_ball.rviz'  # adjust filename if needed
    )

    return LaunchDescription([
        # Launch RViz2 (use config if available)
        ExecuteProcess(
            cmd=[
                'rviz2',
                '-d', rviz_config_path
            ],
            output='screen'
        ),

        # video_publisher node
        Node(
            package='par_catch_ball',
            executable='video_publisher',
            name='video_publisher',
            output='screen',
        ),

        # ball_detector node
        Node(
            package='par_catch_ball',
            executable='ball_detector',
            name='ball_detector',
            output='screen',
        ),

        # trajectory_estimator node
        Node(
            package='par_catch_ball',
            executable='trajectory_estimator',
            name='trajectory_estimator',
            output='screen',
        ),
    ])
