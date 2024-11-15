from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    det_node = Node(
        package='seg_det',
        executable='det_node',
        name='detection_node',
        output='screen',
    )

    cam_node = Node(
        package='seg_det',
        executable='cam_node',
        name='camera_node',
        output='screen',
    )

    return LaunchDescription([
        det_node,
        cam_node,
    ])