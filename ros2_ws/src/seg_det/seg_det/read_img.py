

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        
        # Publisher
        self.publisher_ = self.create_publisher(Image, 'camera/image', 10)
        self.bridge = CvBridge()
        
        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)  # 0 is usually the default camera
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open the webcam")
            return

        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        # Capture a frame from the webcam
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to capture image from webcam")
            return

        # Convert the captured frame to a ROS Image message
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        
        # Publish the image message
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing image from webcam to camera/image')

    def destroy_node(self):
        # Release the webcam when the node is destroyed
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()