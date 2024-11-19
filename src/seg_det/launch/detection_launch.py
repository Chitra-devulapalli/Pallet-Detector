from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    det_node = Node(
        package='seg_det',
        executable='det_node',
        name='detection_node',
        output='screen',
    )

    # cam_node = Node(
    #     package='seg_det',
    #     executable='cam_node',
    #     name='camera_node',
    #     output='screen',
    # )

    rosbag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '/ros2_ws/src/seg_det/seg_det/internship_assignment_sample_bag'],
        name='rosbag_play',
        output='screen',
    )

    return LaunchDescription([
        rosbag_play,
        det_node,
        # cam_node,
    ])