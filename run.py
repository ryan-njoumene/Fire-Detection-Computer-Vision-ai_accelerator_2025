"""Import Essentials Libraries"""
import os
from multiprocessing import freeze_support  # Import freeze_support
import cv2
import yaml
import comet_ml
import numpy as np
import torch, torchvision

"""Import YOLOv11 model and its utilities functions"""
from ultralytics import YOLO
from ultralytics.utils.callbacks.base import add_integration_callbacks
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionPredictor

"""Import Utilities Functions"""
from utilities import monitoring_gpu_usage, MAGENTA_FONT, RED_FONT, BLUE_FONT, RESET_COLOR, CLASSES_TO_DETECT

"""Import Model Training Setting"""
from trainning_settings import DATA, VAL, PROJECT, EXIST_OK, NAME, PLOTS, PROFILE, EPOCHS, PATIENCE, BATCH_SIZE, IMGSZ, CACHE, AMP, WORKERS, SAVE, SAVE_PERIOD, OPTIMIZER, COS_LR, SINGLE_CLS, FOCUS_CLASSES

"""Import Model Augmentation Setting"""
from augmentation_settings import HUE_ADJUSTMENT, SATURATION_ADJUSTMENT, BRIGHTNESS_ADJUSTMENT, GEOMETRIC_TRANSFORMATION, GEOMETRIC_TRANSLATION, SCALE, SHEAR, PERSPECTIVE, FLIP_UPDOWN, FLIP_LEFTRIGHT, BGR_CHANNEL_SWAP, MOSAIC, MIXUP


# ---------------------------------
# ---------------------------------

# PACKAGES TO DOWNLOAD

# pip install comet_ml
# pip install ultralytics
# from comet_ml.integration.pytorch import log_model
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# ---------------------------------
# ---------------------------------

# GPU ACCELERATION SETTINGS

# set GPU as processing Unit for Running our Model
DEVICE = monitoring_gpu_usage()

# ---------------------------------
# ---------------------------------

# TRAINING SETTINGS FOR MODEL

trainings_settings = dict(
                data=DATA,
                device=DEVICE,
                val=VAL, 
                project=PROJECT, 
                exist_ok=EXIST_OK, 
                name=NAME, 
                plots=PLOTS, 
                profile=PROFILE, 
                epochs=EPOCHS, 
                patience=PATIENCE, 
                batch=BATCH_SIZE, 
                # imgsz=IMGSZ, 
                cache=CACHE,
                amp=AMP,
                workers=WORKERS,
                save=SAVE,
                save_period = SAVE_PERIOD,
                optimizer=OPTIMIZER,
                cos_lr=COS_LR,
                single_cls=SINGLE_CLS,
                # classes=FOCUS_CLASSES
                # model=model
                )

# ---------------------------------
# ---------------------------------

# AUGMENTATION SETTINGS FOR MODEL

augmentation_settings = dict(
                    hsv_h=HUE_ADJUSTMENT, 
                    hsv_s=SATURATION_ADJUSTMENT,
                    hsv_v=BRIGHTNESS_ADJUSTMENT,
                    degrees=GEOMETRIC_TRANSFORMATION,
                    translate=GEOMETRIC_TRANSLATION,
                    scale=SCALE,
                    shear=SHEAR,
                    perspective=PERSPECTIVE,
                    flipud=FLIP_UPDOWN,
                    fliplr=FLIP_LEFTRIGHT,
                    bgr=BGR_CHANNEL_SWAP,
                    mosaic=MOSAIC,
                    mixup=MIXUP
                    )

# ---------------------------------
# ---------------------------------

# INITIALIZE A COMET EXPERIMENT

# Set COMET API key, project name, and workspace
comet_api = os.environ["COMET_API_KEY"]
os.environ["COMET_PROJECT_NAME"] = "Fire_Detection_aider128_AI_Accelerator_2025"
os.environ["COMET_WORKSPACE"] = "ryan-njoumene"

# Set the Logging Process of our AI processing to COMET
try:
    comet_ml.login(project_name=os.environ["COMET_PROJECT_NAME"])
except:
    print(f"\n{RED_FONT}----< Invalid Connexion to your COMET Account. Verify your Credidentials >----{RESET_COLOR}")


# ---------------------------------
# ---------------------------------

# LOAD YOLO MODEL
# Specifies the model file for training. Accepts a path to either a .pt pretrained model or a .yaml configuration file. 

if __name__ == '__main__':
    freeze_support()  # Call freeze_support() if your script might be frozen
    # model = YOLO("yolo11n.yaml")  # build a new model from YAML
    # model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    # model = YOLO("best.pt")  # load a pretrained model (recommended for training)

    print(f"\n{MAGENTA_FONT}----< YOLO11n Model Informations >----{RESET_COLOR}")
    model.info()

    # Use GPU acceleration
    model.to(DEVICE)
    print(f"YOLO11n Model : run with {model.device}")

    # Model Settings
    # See Explanations of Settings in trainning_settings.py
    settings = trainings_settings | augmentation_settings

    # ---------------------------------
    # ---------------------------------

    # TRAINING PHASE & VALIDATION PHASE


    print(f"{BLUE_FONT}\n\nSTART >>>\n{RESET_COLOR}")
    print(f"\nType of 'model' variable: {type(model)}")
    model.train(**settings)

    # Resume Interrupted  training
    # results = model.train(resume=True)
# else:
#     print(f"\n{RED_FONT}----< CANNOT CONNECT TO COMET: LOGGING DATA IS IMPOSSIBLE >----{RESET_COLOR}")

print(f"{BLUE_FONT}\n\nEND >>>{RESET_COLOR}")

# ---------------------------------
# ---------------------------------

# TESTING PHASE
# Define the source for testing
# This can be a single image, a directory of images, a video file, or even a webcam feed.
# source = './datasets/aider128/images/test'

# Test the model
def Testing():
    print(f"{MAGENTA_FONT}\n\nTESTING PHASE >>>\n{RESET_COLOR}")
    results = model.predict(source="./datasets/aider128/images/test", visualize=True, augment=True, agnostic_nms=True)

    # Visualize the results
    print(f"{MAGENTA_FONT}\n\nVISUALIZATION PHASE >>>\n{RESET_COLOR}")
    # print("\n\nVisualization Phase...")
    for result in results:
        result.show()

    print(f"{MAGENTA_FONT}\n\nEND >>>\n{RESET_COLOR}")
