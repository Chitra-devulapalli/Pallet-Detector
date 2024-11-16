import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

from cv_bridge import CvBridge
import numpy as np
import cv2
from ultralytics import YOLO
import onnxruntime as ort
from .unet import UNet

import torch
import torch.nn as nn
import numpy as np


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        print("HEHE")
        self.bridge = CvBridge()
        self.segmentation_model = UNet()
        self.s_model = ort.InferenceSession("/ros2_ws/src/seg_det/seg_det/segmentation.onnx")

        #Change to detection.onnx model if necessary
        self.detection_model = YOLO("/ros2_ws/src/seg_det/seg_det/detection.pt", task="detect") 

        #Subscribers
        self.create_subscription(Image, '/camera/image', self.image_callback, 10) #give camera image topic based on the camera used
        
        #Publishers
        self.segmentation_pub = self.create_publisher(Image, '/output/segmentation', 10)
        self.detection_pub = self.create_publisher(Detection2DArray, '/output/detection', 10)
        self.det_image_pub = self.create_publisher(Image, '/output/detection_img', 10)

        self.COLORS = {
            0: [0, 0, 0],       # Background - Black
            1: [0, 0, 255],     # Pallet - Red
            2: [0, 255, 0]      # Ground - Green
        }

    def map_classes_to_colors(self, segmentation_mask):
        #Create an RGB color image based on the segmentation mask
        color_image = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3), dtype=np.uint8)
        for label, color in self.COLORS.items():
            color_image[segmentation_mask == label] = color
        return color_image
    
    def detected_image(self,cv_image, xywh):
        #returns blank image if no bounding boxes are detected
        bb = np.zeros((416, 416, 3), np.uint8)
        for i, box in enumerate(xywh):
            cx, cy, w, h = box
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            #Draw the rectangle
            color = (255, 0, 0)  # Blue color for bounding box
            bb = cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
        return bb 
        
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        #Run detection and segmentation
        detections = self.run_detection(cv_image)
        segmentation_mask = self.run_segmentation(cv_image)

        self.publish_results(cv_image,detections, segmentation_mask)

    def run_detection(self, image):
        #Preprocess the image for detection
        input_tensor = self.preprocess_image(image)

        detections = self.detection_model(input_tensor)
        return detections  

    def run_segmentation(self, image):
        input_tensor = self.preprocess_image(image)
        input_tensor = input_tensor.numpy()
        input_name = self.s_model.get_inputs()[0].name
        segmentation_output = self.s_model.run(None, {input_name: input_tensor})
        segmentation_mask = segmentation_output[0]
        segmentation_mask = np.argmax(segmentation_mask.squeeze(), axis=0)
        segmentation_mask = segmentation_mask.astype(np.uint8)
        return segmentation_mask

    def preprocess_image(self, image):
        resized_image = cv2.resize(image, (416, 416))
        input_tensor = torch.tensor(resized_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

        return input_tensor

    def publish_results(self, cv_image, detections, segmentation_mask):
        #Convert segmentation mask to ROS Image message and publish
        color_image = self.map_classes_to_colors(segmentation_mask)
        #Convert the color image to a ROS Image message
        mask_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="rgb8")
        self.segmentation_pub.publish(mask_msg)

        #Publish detection results in the Detection2DArray format
        detection_msg = Detection2DArray()
        #print(len(detections))
        xywh = detections[0].boxes.xywh
        det_img = self.detected_image(cv_image, xywh.numpy())
        det_img=self.bridge.cv2_to_imgmsg(det_img, encoding="rgb8")
        self.det_image_pub.publish(det_img)
        class_labels = detections[0].boxes.cls 
        confidences = detections[0].boxes.conf  

        for i in range(xywh.shape[0]):
            detection = Detection2D()

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(int(class_labels[i].item()))  # Convert class label to string as expected
            hypothesis.hypothesis.score = confidences[i].item()  # Set confidence score
            detection.results.append(hypothesis)

            #Set bounding box details
            detection.bbox.center.position.x = xywh[i, 0].item()  # x_center
            detection.bbox.center.position.y = xywh[i, 1].item()  # y_center
            detection.bbox.size_x = xywh[i, 2].item()  # width
            detection.bbox.size_y = xywh[i, 3].item()  # height

            #Append the detection to the Detection2DArray
            detection_msg.detections.append(detection)    

        self.detection_pub.publish(detection_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

