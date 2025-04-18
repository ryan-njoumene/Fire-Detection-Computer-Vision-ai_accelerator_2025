"""Imports"""
import sys
import torch


# ---------------------------------
# ---------------------------------

# FONTS COLORS


# AINSI ESCAPE SEQUENCE for color and text formatting in shell interface.
# More at https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
BLUE_FONT = "\033[1;34m"
YELLOW_FONT = "\033[1;33m"
RED_FONT = "\033[1;31m"
MAGENTA_FONT = "\033[1;35m"
RESET_COLOR = "\033[0m"


# ---------------------------------
# ---------------------------------

# CLASSES OF THE MODEL

# Classes correspond to the elements you want to target and detect in your AI Model
CLASSES_TO_DETECT = ["fire", "flame", "wildfire", "forest fire",
                     "blaze", "flare-up",  "burning", "explosion", 
                     "embers", "sparks", "heat", "smolder", 
                     "mushroom cloud", "smoke", "white smoke", "dark smoke",
                     "smog", "glowing", "light", "red light",
                     "white light",  "orange light",  "yellow light", "burning building",
                     "burning vehicle", "burning trees" ]

# ---------------------------------
# ---------------------------------

# GPU MONITORING

def monitoring_gpu_usage():
    """Verify that my NVIDIA GPU is available and used to process efficiently my function"""
    # Check if CUDA is available
    print(f"\n{YELLOW_FONT}----< Device Informations >----{RESET_COLOR}")
    print(f"Python Interpreter: {sys.executable}")
    print(f"Pytorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    # command: nvidia-smi, nvcc --version

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    # Ensure that your Yolov11 function use GPU Acceleration
    return device
