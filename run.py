"""Import YOLOv11 model"""
from ultralytics import YOLO
import torch, torchvision
import comet_ml
# pip install ultralytics
# pip install comet_ml
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# ---------------------------------
# ---------------------------------

# TRAINNING SETTINGS FOR MODEL
from trainning_settings import DATA, VAL, PROJECT, EXIST_OK, NAME, PLOTS, PROFILE, EPOCHS, PATIENCE, BATCH_SIZE, IMGSZ, CACHE, WORKERS, SAVE, SAVE_PERIOD, OPTIMIZER, COS_LR, SINGLE_CLS, FOCUS_CLASSES
from utilities import monitoring_gpu_usage, MAGENTA_FONT, RESET_COLOR

# set GPU as processing Unit for Running our Model
DEVICE = monitoring_gpu_usage()

# ---------------------------------
# ---------------------------------

# Set the Logging Process of our AI processing to COMET
comet_ml.login(project_name="Fire_Detection_aider128_ai_accelerator_2025")


# ---------------------------------
# ---------------------------------

# LOAD MODEL
# Specifies the model file for training. Accepts a path to either a .pt pretrained model or a .yaml configuration file. 

# model = YOLO("yolo11n.yaml")  # build a new model from YAML
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

print(f"\n{MAGENTA_FONT}----< YOLO11n Model Informations >----{RESET_COLOR}")
model.info()

# Use GPU acceleration
model.to(DEVICE)
print(f"YOLO11n Model : run with {model.device}")

# ---------------------------------
# ---------------------------------

# TRAINING PHASE & VALIDATION PHASE
# epoch of learning, nber of time it runs through all the data (affect time of learning and performance)
# imgsz=640 means all image are rescaled to 640x640 px (ensure better processing and result by using uniform image size)
# results = model.train(data=DATA, epochs=EPOCHS, imgsz=640)


# Resume Interrupted  training
# results = model.train(resume=True)


# ---------------------------------
# ---------------------------------

# TESTING PHASE
# Define the source for testing
# This can be a single image, a directory of images, a video file, or even a webcam feed.
source = 'Fire-Detection-Computer-Vision-ai_accelerator_2025/datasets/aider128/images/test'

# Test the model
# results = model.predict(source=source)

# Visualize the results
# print("\n\nVisualization Phase...")
# for result in results:
#     result.show()


# Testing Visual Representation with MaPlotLib
# from PIL import Image
# import matplotlib.pyplot as plt
# image = Image.open('Fire-Detection-Computer-Vision-ai_accelerator_2025/datasets/aider128/images/train/fire_image0001.jpg')
# plt.imshow(image)
# plt.show()