# Object Detection using YOLOv3 and OpenCV

This project demonstrates how to implement real-time object detection using **YOLOv3 (You Only Look Once, Version 3)** and OpenCV in Python. The system identifies and classifies multiple objects in images or video streams, draws bounding boxes around detected objects, and displays confidence scores for each prediction.

## Features

- Detect multiple objects in a single image.
- Identify 80 object classes from the COCO dataset.
- Draw bounding boxes and class labels around detected objects.
- Display confidence scores for detected objects.
- Process images, prerecorded videos, and webcam streams.

## Technologies Used

- Python
- OpenCV
- YOLOv3
- NumPy
- COCO dataset

## How It Works

1. Load the YOLOv3 model configuration and pretrained weights.
2. Read an image or video frame using OpenCV.
3. Convert the input into a blob and pass it through the neural network.
4. Extract predicted object classes, bounding boxes, and confidence scores.
5. Apply Non-Maximum Suppression (NMS) to remove overlapping detections.
6. Display the final detections using OpenCV.

## Applications

- Video surveillance
- Traffic monitoring
- Pedestrian detection
- Smart retail systems
- Object recognition in computer vision applications

## Conclusion

This project provides a practical introduction to deep-learning-based object detection using YOLOv3 and OpenCV. It demonstrates how pretrained neural networks can identify multiple objects efficiently in images and video streams.

pip install opencv-python numpy


Download these YOLOv3 model files and place them in your project directory:
- yolov3.cfg — model configuration
- yolov3.weights — pretrained model weights
- coco.names — object class labels
Then create object_detection.py:
import cv2import numpy as np# Load class labelswith open("coco.names", "r") as file:    classes = [line.strip() for line in file.readlines()]# Load YOLOv3 modelnet = cv2.dnn.readNetFromDarknet(    "yolov3.cfg",    "yolov3.weights")layer_names = net.getLayerNames()output_layers = [    layer_names[i - 1]    for i in net.getUnconnectedOutLayers().flatten()]# Read input imageimage = cv2.imread("image.jpg")if image is None:    raise FileNotFoundError("Could not load image.jpg")height, width = image.shape[:2]# Prepare image for YOLOblob = cv2.dnn.blobFromImage(    image,    scalefactor=1 / 255.0,    size=(416, 416),    swapRB=True,    crop=False)



Run the project:
python object_detection.py
