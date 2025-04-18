"""Import YOLOv11 model"""
from ultralytics import YOLO
import torch, torchvision
import comet_ml
from ultralytics.utils.callbacks.base import add_integration_callbacks
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionPredictor
import torch
import cv2
import numpy as np
import yaml
from multiprocessing import freeze_support  # Import freeze_support
# from comet_ml.integration.pytorch import log_model
import os
# pip install ultralytics
# pip install comet_ml
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# ---------------------------------
# ---------------------------------

# TRAINNING SETTINGS FOR MODEL
from trainning_settings import DATA, VAL, PROJECT, EXIST_OK, NAME, PLOTS, PROFILE, EPOCHS, PATIENCE, BATCH_SIZE, IMGSZ, CACHE, AMP, WORKERS, SAVE, SAVE_PERIOD, OPTIMIZER, COS_LR, SINGLE_CLS, FOCUS_CLASSES
from utilities import monitoring_gpu_usage, MAGENTA_FONT, RED_FONT, BLUE_FONT, RESET_COLOR, CLASSES_TO_DETECT

# set GPU as processing Unit for Running our Model
DEVICE = monitoring_gpu_usage()

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
# is_Connection_to_COMET_Valid = True
# try:
#     experiment = comet_ml.Experiment(
#         api_key=comet_api,
#         project_name=project_name,
#         workspace=workspace,
#         log_system_details=True,  # Ensure system details logging is enabled
#         log_env_gpu=True,        # Specifically enable GPU metrics
#         log_env_cpu=True,        # Specifically enable CPU metrics
#         log_env_host=True,       # Optional: Log host information
#     )

#     # You can also log hyperparameters before training starts
#     focus = None
#     if FOCUS_CLASSES is None: 
#         focus = CLASSES_TO_DETECT
#     hyperparameters = {
#         "model": "yolov11n",
#         "name": NAME,
#         "augmentation" : False,
#         "epochs": EPOCHS,
#         "patience": PATIENCE,
#         "batch_size": BATCH_SIZE,
#         "image_size": IMGSZ,
#         "learning_rate": 0.01, #YOLO default
#         "optimizer": OPTIMIZER,
#         "cosine_learning_rate": COS_LR, 
#         "single_class": SINGLE_CLS, 
#         "focus_classes" : focus,
#         "workers": WORKERS
#     }
#     experiment.log_parameters(hyperparameters)

#     # If you are using a data configuration file (e.g., coco128.yaml), log its name
#     data_config = "data_config/aider128.yaml"
#     experiment.log_parameter("data_config", data_config)

# # Handdle Errors Gracefully
# except Exception as e:
#     print(f"\n{RED_FONT}----< Invalid Connexion to your COMET Account. Verify your Credidentials >----{RESET_COLOR}")
#     print(f"{RED_FONT}{e}{RESET_COLOR}")
#     is_Connection_to_COMET_Valid = False

# # Get training data path from your config
# with open(data_config, 'r') as f:
#     import yaml
#     data_info = yaml.safe_load(f)
#     train_path = data_info.get('train')
#     val_path = data_info.get('val')
#     if train_path:
#         experiment.log_parameter("train_data_path", train_path)
#     if val_path:
#         experiment.log_parameter("val_data_path", val_path)


# ---------------------------------
# ---------------------------------

# LOGGING DATA AND TRAINING METRIC OF THE MODEL WITH COMET_ML

# class CometCallback(BaseTrainer):
#     def on_fit_start(self, trainer):
#         self.experiment = trainer.opt.comet_experiment  # Access the Comet experiment

#     def on_train_epoch_end(self, trainer):
#         epoch = trainer.epoch
#         metrics = trainer.label_loss, trainer.box_loss, trainer.obj_loss, trainer.cls_loss, trainer.l1_loss, trainer.loss
#         names = ['train/label_loss', 'train/box_loss', 'train/obj_loss', 'train/cls_loss', 'train/l1_loss', 'train/loss']
#         for name, value in zip(names, metrics):
#             self.experiment.log_metric(name, value, step=epoch)

#         # Log learning rate
#         lr = trainer.lr[0] if isinstance(trainer.lr, list) else trainer.lr
#         self.experiment.log_metric("learning_rate", lr, step=epoch)

#     def on_val_end(self, trainer):
#         epoch = trainer.epoch
#         metrics = trainer.metrics  # Contains validation metrics like mAP, precision, recall
#         if metrics:
#             self.experiment.log_metric("val/precision", metrics.precision.item(), step=epoch)
#             self.experiment.log_metric("val/recall", metrics.recall.item(), step=epoch)
#             self.experiment.log_metric("val/mAP50", metrics.map50.item(), step=epoch)
#             self.experiment.log_metric("val/mAP50-95", metrics.map.item(), step=epoch)

#         # Log validation predictions (example for the last validation batch)
#         if trainer.validator.pred and trainer.validator.imgs:
#             for i, (im_file, pred) in enumerate(zip(trainer.validator.im_files, trainer.validator.pred)):
#                 if i < 5:  # Log a few example images
#                     orig_img = cv2.imread(im_file)
#                     det_img = trainer.validator.plot_bboxes(pred.cpu().numpy(), orig_img.copy(), names=trainer.model.names)
#                     experiment.log_image(det_img, name=f"val_prediction_epoch_{trainer.epoch}_image_{i}")
    
#     # Seamlessly log your Pytorch model
#     def on_train_end(self, trainer):
#         # Log the trained model weights
#         experiment.log_model(f"yolov11n_{PROJECT}_epoch_{trainer.epoch}.pt", trainer.best)
#         experiment.end()

# # Initialize Comet experiment and pass it to the trainer's opt
# try:
#     opt_override = {
#         'comet_experiment': experiment  # Use the initialized experiment object
#     }
# # Handdle Errors Gracefully
# except Exception as e:
#     print(f"\n{RED_FONT}----< Invalid Connexion to your COMET Account. Verify your Credidentials >----{RESET_COLOR}")
#     print(f"{RED_FONT}{e}{RESET_COLOR}")
#     is_Connection_to_COMET_Valid = False

# ---------------------------------
# ---------------------------------

# LOAD YOLO MODEL
# Specifies the model file for training. Accepts a path to either a .pt pretrained model or a .yaml configuration file. 


if __name__ == '__main__':
    freeze_support()  # Call freeze_support() if your script might be frozen
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

    # See Explanations of Settings in trainning_settings.py
    # if is_Connection_to_COMET_Valid == True:
    print(f"{BLUE_FONT}\n\nSTART >>>\n{RESET_COLOR}")
    print(f"T\nype of 'model' variable: {type(model)}")

    trainnings_settings = dict(data="./data_config/aider128.yaml",
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
                        imgsz=IMGSZ, 
                        cache=CACHE,
                        amp=AMP,
                        workers=WORKERS,
                        save=SAVE,
                        save_period = SAVE_PERIOD,
                        optimizer=OPTIMIZER,
                        cos_lr=COS_LR,
                        single_cls=SINGLE_CLS,
                        classes=FOCUS_CLASSES
                        # model=model
                        )

    # Instantiate the DetectionTrainer, passing the loaded model
    # trainer = DetectionTrainer(cfg=model, overrides=trainnings_settings)
    # add_integration_callbacks(trainer)
    # trainer.train()
    model.train(**trainnings_settings)
    # results = model.train(data=DATA,
    #                     val=VAL, 
    #                     project=PROJECT, 
    #                     exist_ok=EXIST_OK, 
    #                     name=NAME, 
    #                     plots=PLOTS, 
    #                     profile=PROFILE, 
    #                     epochs=EPOCHS, 
    #                     patience=PATIENCE, 
    #                     batch=BATCH_SIZE, 
    #                     imgsz=IMGSZ, 
    #                     cache=CACHE,
    #                     workers=WORKERS,
    #                     save=SAVE,
    #                     save_period = SAVE_PERIOD,
    #                     optimizer=OPTIMIZER,
    #                     cos_lr=COS_LR,
    #                     single_cls=SINGLE_CLS,
    #                     classes=FOCUS_CLASSES,
    #                     callbacks=[CometCallback()], 
    #                     **opt_override)

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
