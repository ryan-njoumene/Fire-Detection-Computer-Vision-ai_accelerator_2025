
# Dependancies to Import ...
# pip install sievedata
# sieve login

import cv2
import sieve
import shutil
import os
import zipfile
import tempfile
import numpy as np

# Global Variables
working_dir_path = "ai_accelerator/pipeline/data_annotation/video_segmentation/"
blue_font = "\033[0;34m"
reset_color = "\033[0m"


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
        cv2.imwrite(f'{working_dir_path}first_frame.png', frame)
    else:
        raise Exception("Failed to read the video; empty or does not exist")

    frame = sieve.File(path=f'{working_dir_path}first_frame.png')
    cap.release()

    return frame

# Step 2 : Extract Bounding Box from IMG or Frame from Video

def get_object_bbox(image: sieve.File, object_name: str):
    """Get Bounding Boxes of all Objects corresponding to your Prompt found in the Image"""
    yolo = sieve.function.get('sieve/yolov8')                # yolo endpoint

    response = yolo.run(                                     # call yolo
        file=image,
        classes=object_name,
        models='yolov8l-world',
    )

    box = response['boxes'][0]                                # most confident bounding box
    bounding_box = [box['x1'],box['y1'],box['x2'],box['y2']]  # parse response into list

    return bounding_box

# Sieve

metadata = sieve.Metadata(
    title="text-to-segment",
    description="Text prompt SAM2 to segment a video or image.",
    readme=open(f"{working_dir_path}README.md").read(),
    image=sieve.File(path=f"{working_dir_path}first_frame.png")
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

def segment(file: sieve.File, object_name: str, return_mp4: bool = False):
    """
    Process a file (IMG or VIDEO) and Return its Segmented Data
    :param file: photo or video to segment
    :param object_name: the object you wish to segment
    :param return_mp4: if True, return only an MP4 video of the segmentation masks
    """
    sam = sieve.function.get("sieve/sam2")

    if is_video(file):
        image = get_first_frame(file)
    else:
        image = file

    print("fetching bounding box...")
    box = get_object_bbox(image, object_name)

    sam_prompt = {
        "object_id": 1,   # id to track the object
        "frame_index": 0, # first frame (if it's a video)
        "box": box        # bounding box [x1, y1, x2, y2]
    }

    sam_out = sam.run(
        file=file,
        prompts=[sam_prompt],
        model_type="tiny",
        debug_masks=return_mp4
    )

    return sam_out

# Convert File of frames to an mp4 Video

def zip_to_mp4(frames_zip: sieve.File):
    """
    convert zip file of frames to an mp4
    """
    output_path = f"{working_dir_path}output_video.mp4"
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(frames_zip.path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        images = [img for img in os.listdir(temp_dir) if img.endswith(".png")]
        images = sorted(images, key=lambda x: int(x.split('_')[1]))

        first_frame = cv2.imread(os.path.join(temp_dir, images[0]))
        height, width, layers = first_frame.shape
        frame_size = (width, height)

        # Define the codec and create VideoWriter object
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, frame_size)

        # Loop through the images and write them to the video
        for image in images:
            img_path = os.path.join(temp_dir, image)
            frame = cv2.imread(img_path)
            out.write(frame)

    out.release()
    return sieve.File(path=output_path)

# Step 4 : Main Method

def Image_to_Segmented_Mask():
    # ex: file_path = "duckling.mp4"
    file_path = "../ai_accelerator/datasets/aider128/images/train/fire_image0002.jpg"
    # ex: text_prompt = "duckling"
    text_prompt = "fire"

    file = sieve.File(path=file_path)
    sam_out = segment(file, text_prompt)

    # os.makedirs("outputs", exist_ok=True)
    os.chdir("ai_accelerator/datasets/aider128/masks/train/")

    # make a image/video file coresponding to the generated mask
    # ex for video: mask = zip_to_mp4(sam_out['masks'])
    mask = sam_out['masks']
    mask_image = (mask * 255).astype(np.uint8)  # Convert to uint8 format
    cv2.imwrite('fire_image0002_mask.jpg', mask_image)
    

    # ex for video: shutil.move(mask.path, "output.mp4")
    # move file to new location and rename it

def Video_to_Segmented_Mask():

    video_filename = "butterfly.mp4"
    video_path = f"{working_dir_path}{video_filename}"

    text_prompt = "butterfly"
    video = sieve.File(path=video_path)
    sam_out = segment(video, text_prompt)

    mask = zip_to_mp4(sam_out['masks'])

    outputs_dir_path = f"{working_dir_path}outputs"
    os.makedirs(outputs_dir_path, exist_ok=True)
    shutil.move(mask.path, os.path.join(outputs_dir_path, f"segment_{video_filename}"))

    print(f"\n{blue_font}TASK COMPLETED!{reset_color}")

Video_to_Segmented_Mask()