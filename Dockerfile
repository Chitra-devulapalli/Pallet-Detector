# Use the ROS Humble base image
FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV COLCON_DEFAULTS_FILE=/home/ros2_ws

WORKDIR /ros2_ws

COPY ./src /ros2_ws/src
COPY ./requirements.txt /ros2_ws/requirements.txt
COPY ./ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh


RUN apt-get update && apt-get install -y \
    ros-humble-sensor-msgs \
    ros-humble-vision-msgs \
    ros-humble-cv-bridge \
    ros-humble-rosbag2-py \
    python3-colcon-common-extensions \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -r /ros2_ws/requirements.txt

RUN bash -c "source /opt/ros/$ROS_DISTRO/setup.bash"

CMD ["bash", "-c", "colcon build && source install/setup.bash && ros2 launch seg_det detection_launch.py"]

# CMD ["bash", "-c", "colcon build && source install/setup.bash"]