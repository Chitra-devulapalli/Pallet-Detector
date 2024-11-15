# Pallet Detector

This project focuses on developing a pallet detection and segmentation application in ROS2 Humble, optimized for edge devices such as the NVIDIA Jetson AGX Orin. It is designed for real-time mobile robotics applications in manufacturing or warehousing environments.

## Overview

### Data Preparation:
- Annotated the provided dataset and split it into training, validation, and test sets.
- Used Grounded SAM to get both bounding box and segmented outputs based on a prompt.

  ![Screenshot from 2024-11-14 23-19-06](https://github.com/user-attachments/assets/a45a873c-3c67-48c9-9bb7-cf081d4601d3)
  ![Screenshot from 2024-11-14 23-19-18](https://github.com/user-attachments/assets/e001e2d5-f3da-4beb-9f08-42d15d896e86)



- Applied data augmentation to simulate real-world scenarios with varied lighting conditions.


### Model Development:
- **YOLOv11 Model**: Trained to detect pallets in images.
- **U-Net Model**: Developed for semantic segmentation to distinguish pallets from the background.
![Screenshot from 2024-11-15 16-29-37](https://github.com/user-attachments/assets/1e12bdea-5351-4dad-a606-1edd7722dc21)


### Model Tuning and Evaluation:
- Evaluated the YOLO model using Mean Average Precision (mAP).
- Evaluated the U-Net model using Intersection over Union (IoU).
- Tested pruning techniques, but it impacted performance negatively. Models were converted to ONNX for optimized deployment.

### ROS2 Node Development:
- **Image Publisher Node**: Captures video from a camera feed and publishes to the `camera/image` topic.
- **Detection and Segmentation Node**: Subscribes to `camera/image` and performs pallet detection and segmentation, publishing results to:
  - **`/output/segmentation`**: Outputs segmentation results as an array.
  - **`/output/detection`**: Publishes detection results in `Detection2D` format. Subscribers to this topic should handle data accordingly.
## todo - add a topic for displaying detections. 
## todo - do display

### Dockerization:
- The application was containerized to run on devices with NVIDIA GPU access, providing portability and ease of deployment.

## Instructions to Run the Application

### 1. Build the Docker Image
Navigate to the directory containing your Dockerfile and run the following command to build the Docker image:

```
docker build -t pallet_detection . 
```

### 2. Run the Docker Container

#### Using a Camera Plugin
If your setup uses a camera plugin: Ensure that the correct camera topic is specified in `cam_subscriber.py` (typically on line 28) to match your plugin’s topic configuration.

#### Without a Camera Plugin
If no plugin is available: Run the Docker container with permissions to access the physical camera using (web cam in my case): 

```
sudo docker run -it --rm --gpus all --device /dev/video0:/dev/video0 --name ros2_pallet_detection pallet_detection
```
