"""Import YOLOv11 model"""
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
# epoch of learning, nber of time it runs through all the data (affect time of learning and performance)
# imgsz=640 means all image are rescaled to 640x640 px (ensure better processing and result by using uniform image size)
results = model.train(data="ai_accelerator/data/aider128.yaml", epochs=1, imgsz=640)

# Validate the model on a different dataset
model.val(data="ai_accelerator/data/aider128.yaml")

# TESTING

# Define the source for testing
# This can be a single image, a directory of images, a video file, or even a webcam feed.
source = 'ai_accelerator/datasets/aider128/images/test'

# Test the model
results = model.predict(source=source)

# Visualize the results
print("\n\nVisualization Phase...")
for result in results:
    result.show()


# Testing Visual Representation with MaPlotLib
# from PIL import Image
# import matplotlib.pyplot as plt
# image = Image.open('ai_accelerator/datasets/aider128/images/train/fire_image0001.jpg')
# plt.imshow(image)
# plt.show()