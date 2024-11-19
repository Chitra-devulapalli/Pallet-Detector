

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2

# class ImagePublisher(Node):
#     def __init__(self):
#         super().__init__('image_publisher')
        
#         # Publisher
#         self.publisher_ = self.create_publisher(Image, 'camera/image', 10)
#         self.bridge = CvBridge()
        
#         # Initialize the webcam
#         self.cap = cv2.VideoCapture(0)  # 0 is usually the default camera
#         if not self.cap.isOpened():
#             self.get_logger().error("Failed to open the webcam")
#             return

#         self.timer = self.create_timer(1.0, self.timer_callback)

#     def timer_callback(self):
#         # Capture a frame from the webcam
#         ret, frame = self.cap.read()
#         if not ret:
#             self.get_logger().error("Failed to capture image from webcam")
#             return

#         # Convert the captured frame to a ROS Image message
#         msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        
#         # Publish the image message
#         self.publisher_.publish(msg)
#         self.get_logger().info('Publishing image from webcam to camera/image')

#     def destroy_node(self):
#         # Release the webcam when the node is destroyed
#         if self.cap.isOpened():
#             self.cap.release()
#         super().destroy_node()

# def main(args=None):
#     rclpy.init(args=args)
#     node = ImagePublisher()
#     try:
#         rclpy.spin(node)
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()


# import subprocess
# import time

# def play_rosbag_in_loop(bag_file_path, loop_interval=1):
#     """
#     Play a ROS2 bag file in a loop using subprocess.
    
#     Args:
#         bag_file_path (str): The path to the ROS2 bag file or directory.
#         loop_interval (int): Time in seconds between each playback.
#     """
#     try:
#         while True:
#             print(f"Playing bag file: {bag_file_path}")
            
#             # Run the ros2 bag play command
#             process = subprocess.run(
#                 ['ros2', 'bag', 'play', bag_file_path],
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE
#             )
            
#             # Check for errors
#             if process.returncode != 0:
#                 print(f"Error while playing the bag file:\n{process.stderr.decode()}")
#                 break

#             print("Bag file playback completed.")
            
#             # Wait for the specified interval before replaying
#             time.sleep(loop_interval)
#     except KeyboardInterrupt:
#         print("Bag playback loop interrupted by user.")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     # Path to your ROS2 bag file or directory
#     bag_file_path = "/ros2_ws/src/seg_det/seg_det/internship_assignment_sample_bag"
#     loop_interval = 2  # Time in seconds between replays

#     play_rosbag_in_loop(bag_file_path, loop_interval)
