from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'par_catch_ball'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, "urdf"), glob('urdf/*')),
        (os.path.join('share', package_name, "srdf"), glob('srdf/*')),
        (os.path.join('share', package_name, "config"), glob('config/*')),
        (os.path.join('share', package_name, "rviz"), glob('rviz/*')),
        (os.path.join('share', package_name, "objects"), glob('objects/*')),
        (os.path.join('share', package_name, "meshes/visual"), glob('meshes/visual/*')),
        (os.path.join('share', package_name, "meshes/collision"), glob('meshes/collision/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rosuser',
    maintainer_email='rosuser@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ball_detector= par_catch_ball.ball_detector:main',
            'trajectory_estimator= par_catch_ball.trajectory_estimator:main',
            'interception_planner= par_catch_ball.interception_planner:main',
            'arm_controller= par_catch_ball.arm_controller:main',
        ],
    },
)
