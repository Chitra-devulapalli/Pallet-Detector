# Pallet Detector

This project focuses on developing a pallet detection and segmentation application in ROS2 Humble, optimized for edge devices such as the NVIDIA Jetson AGX Orin. It is designed for real-time mobile robotics applications in manufacturing or warehousing environments.

## Overview

### Data Preparation:
- Annotated the provided dataset and split it into training, validation, and test sets.
- Applied data augmentation to simulate real-world scenarios with varied lighting conditions.

### Model Development:
- **YOLO Model**: Trained to detect pallets in images.
- **U-Net Model**: Developed for semantic segmentation to distinguish pallets from the background.

### Model Tuning and Evaluation:
- Evaluated the YOLO model using Mean Average Precision (mAP).
- Evaluated the U-Net model using Intersection over Union (IoU).
- Tested pruning techniques, but it impacted performance negatively. Models were converted to ONNX for optimized deployment.

### ROS2 Node Development:
- **Image Publisher Node**: Captures video from a camera feed and publishes to the `camera/image` topic.
- **Detection and Segmentation Node**: Subscribes to `camera/image` and performs pallet detection and segmentation, publishing results to:
  - **`/output/segmentation`**: Outputs segmentation results as an array.
  - **`/output/detection`**: Publishes detection results in `Detection2D` format. Subscribers to this topic should handle data accordingly.

### Dockerization:
- The application was containerized to run on devices with NVIDIA GPU access, providing portability and ease of deployment.

## Instructions to Run the Application

### 1. Build the Docker Image
Navigate to the directory containing your Dockerfile and run the following command to build the Docker image:


```bash
docker build -t pallet_detection .  

### 2. Run the Docker Container

#### Using a Camera Plugin
If your setup uses a camera plugin: Ensure that the correct camera topic is specified in `cam_subscriber.py` (typically on line 28) to match your plugin’s topic configuration.

#### Without a Camera Plugin
If no plugin is available: Run the Docker container with permissions to access the physical camera using:

```bash
sudo docker run -it --rm --gpus all --privileged --device /dev/video0:/dev/video0 --name ros2_pallet_detection pallet_detection