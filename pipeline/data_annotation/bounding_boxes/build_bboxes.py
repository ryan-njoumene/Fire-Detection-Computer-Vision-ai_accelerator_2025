
# Dependancies to Import ...
# pip install sievedata
# sieve login

import cv2
import sieve
import os
import numpy as np
import torch
import sys

# Global Variables

DATASET_IMG_TRAINING_PATH = "ai_accelerator/datasets/aider128/images/train/"
DATASET_IMG_VALIDATING_PATH = "ai_accelerator/datasets/aider128/images/val/"

DATASET_LABELS_TRAINING_PATH = "ai_accelerator/datasets/aider128/labels/train/"
DATASET_LABELS_VALIDATING_PATH = "ai_accelerator/datasets/aider128/labels/val/"

WORKING_DIR_PATH = "ai_accelerator/pipeline/data_annotation/bounding_boxes/"
BLUE_FONT = "\033[0;34m"
RESET_COLOR = "\033[0m"


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
    yolo = sieve.function.get('sieve/yolov8')                # yolo endpoint

    response = yolo.run(
        # file=sieve.File(path=image.path)
        # doing this is wrong because sieve module raise a warning when processing
        # the same sieve.File multiple time. The sieve-client library might internally
        # associate some state with a sieve.File object after it's been used in a run call.
        # When you reuse the same sieve.File object in subsequent calls, this existing internal
        # state might be causing a conflict or an attempt to re-initialize something that
        # should only be done once.

        # to solve this solution, simply create a new sieve.File that point at the same image
        file=sieve.File(path=image.path),           # call yolo
        classes=object_name,
        models='yolov8l-world',
    )

    box = response['boxes'][0]                                # most confident bounding box
    bounding_box = [box['x1'],box['y1'],box['x2'],box['y2']]    # parse response into list
    return bounding_box

# Sieve Function and Metadata

metadata = sieve.Metadata(
    title="text-to-segment",
    description="Text prompt SAM2 to segment a video or image.",
    readme= open(f"{WORKING_DIR_PATH}README.md", encoding="utf-8").read(),
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
    class_index = 0

    for object_name in prompts:
        print(f"Fetching bounding box for [{object_name}]...\n")
        # bbox_format = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
        box = get_object_bbox(image, object_name)
        # add the class_index corresponding to the prompted object
        # FORMAT expected by YOLO model: class_index x_center y_center width_bbox height_bbox
        box.append(class_index)

        bbox_all_object[object_name] = box
        class_index += 1

    # return all bbox corresponding to each prompt made
    return bbox_all_object



# Step 4 : Get Dimension of the IMG currently processed

def get_image_dimension(file_path: str):
    """Read Properties of an Image and return its Dimensions"""
    image_properties = cv2.imread(file_path).shape
    img_dimension = {"img_width": image_properties[0], "img_height": image_properties[1]}
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
        print(f"\n{label_filename} created. Proceed to append bbox...")

    except FileExistsError:
        print(f"\n{label_filename} already exists. Proceed to overwrite bbox...")
        img_label = open(f"{path_to_labels}{label_filename}", "w", encoding="utf-8")
        img_label.write("")
        img_label.close()

# Step 7 : Populate .txt file with bbox coordinates corresponding to elements of interests in IMG (model classes)
def fill_label_file_with_bbox_coordinates(path_to_labels: str, label_filename: str, img_dimension: {}, bbox_all_object: {}):
    """Write bounding box coordinates inside the previously created file"""
    lines = []
    for bbox_from_prompt in bbox_all_object.items():
        print(f"prompt: {bbox_from_prompt}")
        bbox = bbox_from_prompt[1]
        label = bbox_to_label_format(bbox, img_dimension)
        lines.append(f"{label["class_index"]} {label["x_center"]} {label["y_center"]} {label["width"]} {label["height"]}\n")

    img_label = open(f"{path_to_labels}{label_filename}", "a", encoding="utf-8")
    img_label.writelines(lines)
    img_label.close()

# Step 8 : transform the Bounding box informations in the Format used for labelling expected by Yolo
def bbox_to_label_format(bbox: [], img_dimension: {}):
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

# Step 9 : Coordinate all the previous functions to detect (prompted elements), generate (bbox coordinate) and write bounding boxes (in .txt file)
def build_bbox_for_img(filename: str, path_to_dataset: str, path_to_labels: str):
    """Create .txt file with bounding boxes of all prompted object by processing an IMG file"""
    file_path = f"{path_to_dataset}{filename}"
    img_dimension = get_image_dimension(file_path)

    text_prompt = ["fire", "smoke", "explosion", "mushroom cloud"]
    file = sieve.File(path=file_path)

    bbox_all_object = generate_bbox_according_to_prompt(file, text_prompt)
    name = remove_extension_from_filename(filename)
    label_filename = f"{name}.txt"

    create_label_file(path_to_labels, label_filename)
    fill_label_file_with_bbox_coordinates(path_to_labels, label_filename, img_dimension, bbox_all_object)
    print(f"\n{BLUE_FONT}{label_filename} COMPLETED!{RESET_COLOR}")



# Main : Build Bounding Boxes .txt file for all Images of a dataset
def create_bounding_boxes(file_location: str):
    """Create Bounding Boxes for all Images of a Training dataSet or Validation dataSet"""

    if file_location == "training":
        path_to_dataset = DATASET_IMG_TRAINING_PATH
        path_to_labels = DATASET_LABELS_TRAINING_PATH
    elif file_location == "validation":
        path_to_dataset = DATASET_IMG_VALIDATING_PATH
        path_to_labels = DATASET_LABELS_VALIDATING_PATH
    else:
        raise ValueError("Invalid file location")
    
    monitoring_gpu_usage()
    
    print(f"\n{BLUE_FONT}Creating Bounding Boxes for {file_location.upper()} Dataset...{RESET_COLOR}\n")
    loop_over_all_file(path_to_dataset, path_to_labels)
    print(f"\n{BLUE_FONT}TASK COMPLETED!{RESET_COLOR}")


def loop_over_all_file(path_to_dataset: str, path_to_labels: str):
    """Go trough each file of a specific directory, get their filename and create their respective bbox label .txt file"""
    # file_cursor = os.scandir(path_to_dataset)
    # for file in file_cursor:
    #     if file.is_file():
    #         filename = file.name

    run_in_silent_mode("fire_image0002.jpg", path_to_dataset, path_to_labels)
    # build_bbox_for_img("fire_image0002.jpg", path_to_dataset, path_to_labels)
    # file_cursor.close()

# Extra : Run In Silent Mode (No message, output or error displayed)
def run_in_silent_mode(filename: str, path_to_dataset: str, path_to_labels: str):
    """Run the current process without outputing any message in terminal"""
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    try: 
        build_bbox_for_img(filename, path_to_dataset, path_to_labels)
    finally:
        sys.stdout = sys.__stdout__

# Extra : GPU Monitoring
def monitoring_gpu_usage():
    """Verify that my NVIDIA GPU is available and used to process efficiently my function"""
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

#Run
create_bounding_boxes("training")




