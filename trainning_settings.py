
# MODEL SETTINGS
# ---------------------------------
# ---------------------------------

# Path to the dataset configuration file (e.g., coco8.yaml). 
# This file contains paths to training and validation data, class names, and number of classes.
DATA = "./data_config/aider128.yaml"

# Enables validation during training, allowing for periodic evaluation of model performance on a separate dataset.
# NO NEED TO RUN .val() AFTERWARD BECAUSE IT IS ALREADY DONE DURING TRAINING BETWEEN EACH EPOCH
VAL = True #default=true

# ---------------------------------
# ---------------------------------

# Name of the project directory where training outputs are saved.
# Allows for organized storage of different experiments.
project_size = {"Nano": "n", "Large": "L"}
bbox_Confidance_Threshold = {"None": "cfNone", "50%": "cf50"}
augmented = {"true": "A", "false": ""}
PROJECT = f"aider128{project_size["Nano"]}_{augmented["false"]}{bbox_Confidance_Threshold["None"]}"

# If True, allows overwriting of an existing project/name directory.
# Useful for iterative experimentation without needing to manually clear previous outputs.
EXIST_OK = False #default=false

# Name of the training run. Used for creating a subdirectory within the project folder, where training logs and outputs are stored.
experiment_numbers = 1
NAME = f"aider128_run{experiment_numbers}"

# ---------------------------------
# ---------------------------------

# Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual
# insights into model performance and learning progression.
PLOTS = True #default=False

# It will export your trained PyTorch model to the ONNX format (+ TensorRT if available on computer)
# ONNX is an open standard for representing machine learning models, allowing them to be run across various frameworks and hardware.
PROFILE = True #default=False

# ---------------------------------
# ---------------------------------

# Total number of training epochs. Each epoch represents a full pass over the entire dataset.
# Adjusting this value can affect training duration and model performance.
EPOCHS = 10 #default=100

# Maximum training time in hours. If set, this overrides the epochs argument,
# allowing training to automatically stop after the specified duration.
# TIME = 3 #default=None

# Number of epochs to wait without improvement in validation metrics before early stopping the training.
# Helps prevent overfitting by stopping training when performance plateaus.
PATIENCE = 10 #default=100

# Number of Images processed simultaneously in a forward pass 
# (data transmitted through layers of the neural network in one instance)
# The batch argument can be configured in three ways:
    # Fixed Batch Size: Set an integer value (e.g., batch=16), specifying the number of images per batch directly.
    # Auto Mode (60% GPU Memory): Use batch=-1 to automatically adjust batch size for approximately 60% CUDA memory utilization.
    # Auto Mode with Utilization Fraction: Set a fraction value (e.g., batch=0.70) to adjust batch size based on the specified fraction of GPU memory usage.
BATCH_SIZE = -1 #default=16

# Target image size for training. All images are resized to this dimension before being fed into the model.
# Affects model accuracy and computational complexity.
IMGSZ = 640 #default=640

# ---------------------------------
# ---------------------------------

# Enables caching of dataset images in memory (True/ram), on disk (disk), or disables it (False).
# Improves training speed by reducing disk I/O at the cost of increased memory usage.
CACHE = True #default=false

# Number of worker threads for data loading.
# Influences the speed of data preprocessing and feeding into the model, especially useful in multi-GPU setups.
WORKERS = 8 #default=8

# ---------------------------------
# ---------------------------------

# Enables saving of training checkpoints and final model weights.
# Useful for resuming training or model deployment.
SAVE = True #default=false
# Frequency of saving model checkpoints, specified in epochs. A value of -1 disables this feature.
# Useful for saving interim models during long training sessions.
SAVE_PERIOD = 3 #default=-1

# ---------------------------------
# ---------------------------------

# Choice of optimizer for training. Options include "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp" etc.,
# or auto for automatic selection based on model configuration. Affects convergence speed and stability.
OPTIMIZER = "auto" #default=auto

# Utilizes a cosine learning rate scheduler, adjusting the learning rate following a cosine curve over epochs.
# Helps in managing learning rate for better convergence.
COS_LR = True #default=False


# ---------------------------------
# ---------------------------------

# Treats all classes in multi-class datasets as a single class during training.
# Useful for binary classification tasks or when focusing on object presence rather than classification.
SINGLE_CLS = False #default=false

# Specifies a list of class IDs to train on. Useful for filtering out and focusing only on certain classes during training.
FOCUS_CLASSES = None #default=None




