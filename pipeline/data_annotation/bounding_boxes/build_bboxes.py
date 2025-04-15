
# Dependancies to Import ...
# pip install sievedata
# sieve login
"""Import Dependancies ..."""
import os
import sys
import cv2 as cv2
import sieve
# import numpy
import torch
from tqdm import trange

# Global Variables

# this file is runned in terminal by the command `python build_bboxes.py`
# which allow us to use CUDA GPU Acceleration from my Nvidia Card
RUN_BY_COMMAND = True

# move from current working directory toward Fire-Detection-Computer-Vision-ai_accelerator_2025 directory
PREFIX_FOR_RELATIVE_PATH = "../../../"
# when play button on vscode is used to run the code, the current
# working dir is set according to where the workspace was open ex: AI_ACELERATOR
PREFIX_FOR_PROJECT_WORKSPACE_PATH = "Fire-Detection-Computer-Vision-ai_accelerator_2025"

if RUN_BY_COMMAND is True:
    PREFIX = PREFIX_FOR_RELATIVE_PATH
else:
    PREFIX = PREFIX_FOR_PROJECT_WORKSPACE_PATH

DATASET_IMG_TRAINING_PATH = f"{PREFIX}/datasets/aider128/images/train/"
DATASET_IMG_VALIDATING_PATH = f"{PREFIX}/datasets/aider128/images/val/"

DATASET_LABELS_TRAINING_PATH = f"{PREFIX}/datasets/aider128/labels/train/"
DATASET_LABELS_VALIDATING_PATH = f"{PREFIX}/datasets/aider128/labels/val/"

WORKING_DIR_PATH = f"{PREFIX}/pipeline/data_annotation/bounding_boxes/"

BLUE_FONT = "\033[1;34m"
YELLOW_FONT = "\033[1;33m"
RED_FONT = "\033[1;31m"
MAGENTA_FONT = "\033[1;35m"
RESET_COLOR = "\033[0m"

# Classes correspond to the elements you want to target and detect in your AI Model
CLASSES_TO_DETECT = ["fire", "flame", "wildfire", "forest fire",
                     "blaze", "flare-up",  "burning", "explosion", 
                     "embers", "sparks", "heat", "smolder", 
                     "mushroom cloud", "smoke", "white smoke", "dark smoke",
                     "smog", "glowing", "light", "red light",
                     "white light",  "orange light",  "yellow light", "burning building",
                     "burning vehicle", "burning trees" ]


# Step 1 : Handling Video Format

def is_video(file: sieve.File):
    """Take a File and Verify if its Extension correpond to a Video Format"""
    file_path = file.path

    video_formats = ['mp4', 'avi', 'mov', 'flv', 'wmv', 'webm', 'mkv']

    if file_path.split(".")[-1] in video_formats:
        return True

    return False


def get_first_frame(video: sieve.File):
    """Extract the First Frame from a Video (Video => to Image)"""
    video_path = video.path

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if ret:
        cv2.imwrite(f'{WORKING_DIR_PATH}first_frame.png', frame)
    else:
        raise FileNotFoundError("Failed to read the video; empty or does not exist")

    frame = sieve.File(path=f'{WORKING_DIR_PATH}first_frame.png')
    cap.release()

    return frame

# Step 2 : Extract Bounding Box from IMG or Frame from Video

def get_object_bbox(image: sieve.File, object_name: str):
    """Get Bounding Boxes of all Objects corresponding to your Prompt found in the Image"""
    yolo = sieve.function.get('sieve/yolov8')       # yolo endpoint

    # More information about Sieve/Yolov8 Function and its Customization at https://www.sievedata.com/functions/sieve/yolov8/guide
    response = yolo.run(                            # call yolo 
        # => file=sieve.File(path=image.path)

        # doing this is wrong because sieve module raise a warning when processing
        # the same sieve.File multiple time. The sieve-client library might internally
        # associate some state with a sieve.File object after it's been used in a run call.
        # When you reuse the same sieve.File object in subsequent calls, this existing internal
        # state might be causing a conflict or an attempt to re-initialize something that
        # should only be done once.

        # to solve this solution, simply create a new sieve.File that point at the same image
        file=sieve.File(path=image.path),
        classes=object_name,
        models='yolov8l-world',

        # ****** confidence_threshold = 0.50 ******
        # The confidence_threshold acts as a filter.
        # Only bounding boxes with a confidence score equal to or greater than this threshold will
        # be considered valid detections and will be outputted by the model.
        # Bounding boxes with confidence scores below the threshold are discarded.
    )

    results = response['boxes']
    if len(results) > 0:
        box = results[0]                                # most confident bounding box
        bounding_box = [box['x1'],box['y1'],box['x2'],box['y2']]    # parse response into list
        return bounding_box
    else:
        return None

# Sieve Function and Metadata

def get_readme():
    """Change the path of the file according to if the code is RUN_BY_COMMAND """
    if RUN_BY_COMMAND is True:
        readme= open("README.md", encoding="utf-8").read()
    else:
        readme= open(f"{WORKING_DIR_PATH}README.md", encoding="utf-8").read()
    return readme

metadata = sieve.Metadata(
    title="text-to-segment",
    description="Text prompt SAM2 to segment a video or image.",
    readme= get_readme()
    # image=sieve.File(path=f"{WORKING_DIR_PATH}first_frame.png")
)


@sieve.function(
    name="text-to-segment",
    python_packages=["opencv-python"],
    system_packages=[
        "libgl1-mesa-glx",
        "libglib2.0-0"
    ],
    metadata=metadata
)


# Step 3 : Segmenting IMG or Video

def generate_bbox_according_to_prompt(file: sieve.File, prompts: list[str]):
    """
    Process a file (IMG or VIDEO) and Return its Segmented Data
    :param file: photo or video to segment
    :param object_name: the object you wish to segment
    """

    if is_video(file):
        image = get_first_frame(file)
    else:
        image = file

    bbox_all_object = {}

        # print(f"\nFetching bounding box for [{object_name}]...\n")
    with trange(len(prompts), colour="yellow") as t:
        for i in t:
            class_index = i
            object_name = prompts[i]
            t.set_description(f"{MAGENTA_FONT}Fetching bounding box for{RESET_COLOR} {RED_FONT}[{object_name}]{RESET_COLOR}")
            # silence any output from sieve function call of get_object_bbox(image, object_name)
            box = run_in_silent_mode(image, object_name)

            # if CLASSES_TO_DETECT not found (set as None) skip to next, else append box
            if box is not None:
                # add the class_index corresponding to the prompted object
                # FORMAT expected by YOLO model: class_index x_center y_center width_bbox height_bbox
                box.append(class_index)

            bbox_all_object[object_name] = box

    # return all bbox corresponding to each prompt made
    return bbox_all_object

# Extra : Run In Silent Mode (No message, output or error displayed)
def run_in_silent_mode(image: sieve.File, object_name: str):
    """Run the current process without outputing any message in terminal"""
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    try:
        # bbox_format = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
        box = get_object_bbox(image, object_name)
    finally:
        sys.stdout = sys.__stdout__
    # return box from get_object_bbox()
    return box


# Step 4 : Get Dimension of the IMG currently processed

def get_image_dimension(file_path: str):
    """Read Properties of an Image and return its Dimensions"""
    image_properties = cv2.imread(file_path).shape
    # .shape returns (Height, Width, Channel), More about OpenCV https://note.nkmk.me/en/python-opencv-pillow-image-size/
    img_dimension = {"img_height": image_properties[0], "img_width": image_properties[1]}
    return img_dimension

# Step 5 : Get Dimension of the IMG currently processed
def remove_extension_from_filename(filename: str):
    """Remove the Extension of a filename"""
    index_start_of_extension = filename.find(".")
    name = filename[0:index_start_of_extension]
    return name

# Step 6 : Creating .txt file containg bbox coordinates
def create_label_file(path_to_labels: str, label_filename: str):
    """Create empty .txt file that will later hold bbox coordinates"""
    try:
        img_label = open(f"{path_to_labels}{label_filename}", "x", encoding="utf-8")
        print(f"\n{YELLOW_FONT}{label_filename} created. Proceed to append bbox...{RESET_COLOR}")

    except FileExistsError:
        print(f"\n{YELLOW_FONT}{label_filename} already exists. Proceed to overwrite bbox...{RESET_COLOR}")
        img_label = open(f"{path_to_labels}{label_filename}", "w", encoding="utf-8")
        img_label.write("")
        img_label.close()

# Step 7 : Populate .txt file with bbox coordinates corresponding
# to elements of interests in IMG (model classes)
def fill_label_file_with_bbox_coordinates(path_to_labels: str, label_filename: str, img_dimension: {int, int}, bbox_all_object: {str, int}):
    """Write bounding box coordinates inside the previously created file"""
    lines = []
    for bbox_from_prompt in bbox_all_object.items():
        print(f"{MAGENTA_FONT}prompt: {RESET_COLOR}{bbox_from_prompt}")
        bbox = bbox_from_prompt[1]
        # if CLASSES_TO_DETECT not found (set as None) skip to next, else append bbox
        if bbox is not None:
            label = bbox_to_label_format(bbox, img_dimension)
            lines.append(f"{label["class_index"]} {label["x_center"]} {label["y_center"]} {label["width"]} {label["height"]}\n")

    img_label = open(f"{path_to_labels}{label_filename}", "a", encoding="utf-8")
    img_label.writelines(lines)
    img_label.close()

# Step 8 : transform the Bounding box informations in the Format used for labelling expected by Yolo
def bbox_to_label_format(bbox: list[int], img_dimension: {str, int}):
    """Normalizes Coordinates (between 0-1) and Set Class_index according to labels format expected by Yolov11"""
    # FORMAT: class_index x_center y_center width height
    x_center = format(bbox[0]/img_dimension["img_width"], '.6f')
    y_center = format(bbox[1]/img_dimension["img_height"], '.6f')

    width_bbox = format(bbox[2]/img_dimension["img_width"], '.6f')
    height_bbox = format(bbox[3]/img_dimension["img_height"], '.6f')

    label_format = {"class_index": bbox[4],
                "x_center": x_center, "y_center": y_center,
                "width": width_bbox, "height": height_bbox}
    # return dictionary of bbox coordinates in the format expected by Yolo
    return label_format

# Step 9 : Coordinate all the previous functions to detect (prompted elements),
# generate (bbox coordinate) and write bounding boxes (in .txt file)
def build_bbox_for_img(filename: str, path_to_dataset: str, path_to_labels: str):
    """Create .txt file with bounding boxes of all prompted object by processing an IMG file"""
    file_path = f"{path_to_dataset}{filename}"
    img_dimension = get_image_dimension(file_path)

    text_prompt = CLASSES_TO_DETECT
    file = sieve.File(path=file_path)

    bbox_all_object = generate_bbox_according_to_prompt(file, text_prompt)
    name = remove_extension_from_filename(filename)
    label_filename = f"{name}.txt"

    create_label_file(path_to_labels, label_filename)
    fill_label_file_with_bbox_coordinates(path_to_labels, label_filename, img_dimension, bbox_all_object)
    print(f"\n{MAGENTA_FONT}{label_filename} BBOXES COMPLETED!{RESET_COLOR}")



# Main : Build Bounding Boxes .txt file for all Images of a dataset
def create_bounding_boxes(file_location: str):
    """Create Bounding Boxes for all Images of a Training dataSet or Validation dataSet"""

    if file_location == "training":
        path_to_dataset = DATASET_IMG_TRAINING_PATH
        path_to_labels = DATASET_LABELS_TRAINING_PATH
        num_iterations = 70
    elif file_location == "validation":
        path_to_dataset = DATASET_IMG_VALIDATING_PATH
        path_to_labels = DATASET_LABELS_VALIDATING_PATH
        num_iterations = 30
    else:
        raise ValueError("Invalid file location")
    monitoring_gpu_usage()
    
    print(f"\n{BLUE_FONT}Creating Bounding Boxes for {file_location.upper()} Dataset...{RESET_COLOR}\n")
    loop_over_all_file(path_to_dataset, path_to_labels, file_location, num_iterations)
    print(f"\n{BLUE_FONT}BOUNDING BOXES {file_location.upper()} DATASET COMPLETED!{RESET_COLOR}")


def loop_over_all_file(path_to_dataset: str, path_to_labels: str, file_location: str, num_iterations: int):
    """Go trough each file of a specific directory, get their filename and create their respective bbox label .txt file"""
    file_cursor = os.scandir(path_to_dataset)

    # trange is a loading_bar from tqdm module
    with trange(len(num_iterations), colour="green") as tm:
        for i in tm:
            tm.set_description(f"Creating Bounding Box for {file_location.capitalize()} Dataset")
            file = file_cursor.__next__()
            if file.is_file():
                filename = file.name
                build_bbox_for_img(filename, path_to_dataset, path_to_labels)

    file_cursor.close()
    # build_bbox_for_img("fire_image0003.jpg", path_to_dataset, path_to_labels)

# Extra : GPU Monitoring
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

    # Ensure that your Sieve/Yolov8 function use GPU Acceleration
    
    # For pre-built functions like sieve/yolov8, Sieve AI likely configures them to run on appropriate hardware,
    # # which for computationally intensive tasks like YOLOv8, would typically include GPUs. You likely don't need to
        # do any specific configuration in your client code.

        # If You Were Deploying a Custom YOLOv8 Function (Not using sieve/yolov8):
        # If you were deploying your own function that runs a YOLOv8 model, you would have more direct control over the environment.
        #  # In that scenario, you would typically:
        # - Specify GPU Resources in Metadata (if Sieve AI allows): Check the Sieve AI documentation for options to define resource
            # requirements (like GPU type or count) in your function's metadata during deployment.
            # - Ensure GPU Dependencies: Your function's environment would need the necessary dependencies for GPU usage
            # (e.g., a base image with NVIDIA drivers and CUDA, and installing GPU-enabled PyTorch or TensorFlow).
            # - Write Code for GPU Utilization: Your Python code within the function would explicitly move the YOLOv8 model and input data
            # to the GPU using the framework's commands (e.g., model.to('cuda') in PyTorch).

#Run (Confidence Treshold from output of Sieve/Yolov8 can be adjusted to have a more precise results)
create_bounding_boxes("training")

# monitoring_gpu_usage()
