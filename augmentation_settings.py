# AUGMENTATIION SETTINGS

# ---------------------------------
# ---------------------------------

# Data augmentation is a crucial technique in computer vision that artificially expands your training dataset by applying 
# various transformations to existing images. When training deep learning models like Ultralytics YOLO, 
# data augmentation helps improve model robustness, reduces overfitting, and enhances generalization to real-world scenarios.

# - Expanded Dataset: By creating variations of existing images, you can effectively increase your training dataset size without collecting new data.
# - Improved Generalization: Models learn to recognize objects under various conditions, making them more robust in real-world applications.
# - Reduced Overfitting: By introducing variability in the training data, models are less likely to memorize specific image characteristics.
# - Enhanced Performance: Models trained with proper augmentation typically achieve better accuracy on validation and test sets.

# More Informations at https://docs.ultralytics.com/guides/yolo-data-augmentation/#using-a-configuration-file

# ---------------------------------
# ---------------------------------

# pip install albumentations
# If the albumentations package is installed, Ultralytics automatically applies a set of extra image augmentations using it.
# These augmentations are handled internally and require no additional configuration.

# Albumentations offers over 70 different transformations, including geometric changes (e.g., rotation, flipping),
# color adjustments(e.g., brightness, contrast), and noise addition (e.g., Gaussian noise).
# Having multiple options enables the creation of highly diverse and robust training datasets.

# More about Albumentations at https://docs.ultralytics.com/integrations/albumentations/

# ---------------------------------
# ---------------------------------

# Adjusts the hue of the image by a fraction of the color wheel, introducing color variability.
# Helps the model generalize across different lighting conditions.
# AKA hsv_h, Range: 0.0 - 1.0, Default: 0.015
HUE_ADJUSTMENT = 0.10 #default=0.015

# Alters the saturation of the image by a fraction, affecting the intensity of colors.
# Useful for simulating different environmental conditions.
# Range : 0.0 - 1.0
SATURATION_ADJUSTMENT = 1.0 #default=0.7

# Alters the saturation of the image by a fraction, affecting the intensity of colors.
# Useful for simulating different environmental conditions.
# Range : 0.0 - 1.0
BRIGHTNESS_ADJUSTMENT = 0.6 #default=0.4

# ---------------------------------
# ---------------------------------

# Rotates the image randomly within the specified degree range, improving the model's 
# ability to recognize objects at various orientations.
# Range : 0.0 - 180
GEOMETRIC_TRANSFORMATION = 180 #default=0.0
# Purpose: Crucial for applications where objects can appear at different orientations.F
# or example, in aerial drone imagery, vehicles can be oriented in any direction, 
# requiring models to recognize objects regardless of their rotation.

# Translates the image horizontally and vertically by a fraction of the image size,
# aiding in learning to detect partially visible objects.
# Range : 0.0 - 1.0
GEOMETRIC_TRANSLATION = 0.25 #default=0.1
# Purpose: Helps models learn to detect partially visible objects and improves robustness to object position. 
# the translation augmentation will teach the model to recognize these features regardless of their completeness or position.

# ---------------------------------
# ---------------------------------

# Scales the image by a gain factor, simulating objects at different distances from the camera.
SCALE = 0.5 #default=0.5

# Shears the image by a specified degree, mimicking the effect of objects being viewed from different angles.
# Range: -180 to +180
SHEAR = 10 #default=0.0
# Purpose: Helps models generalize to variations in viewing angles caused by slight tilts or oblique viewpoints.
# For instance, in traffic monitoring, objects like cars and road signs may appear slanted due to non-perpendicular camera placements.


# Applies a random perspective transformation to the image, enhancing the model's ability to understand objects in 3D space.
# Range: 0.0 - 0.001
PERSPECTIVE = 0.0005 #default=0.0
# Purpose: Perspective augmentation is crucial for handling extreme viewpoint changes, especially in scenarios where objects appear
# foreshortened or distorted due to perspective shifts. For example, in drone-based object detection, buildings, roads, and vehicles
# can appear stretched or compressed depending on the drone's tilt and altitude. By applying perspective transformations, models learn
# to recognize objects despite these perspective-induced distortions, improving their robustness in real-world deployments.

# ---------------------------------
# ---------------------------------

# Flips the image upside down with the specified probability, increasing the data variability without affecting the object's characteristics.
# Range: 0.0 - 1.0,  % chances of being applied
# 1/5 chance of being flip up-or-dawn
FLIP_UPDOWN = 0.20 #default=0.0

# Flips the image upside down with the specified probability, increasing the data variability without affecting the object's characteristics.
# Range: 0.0 - 1.0,  % chances of being applied
# 1/2 chance of being flip left-or-right
FLIP_LEFTRIGHT = 0.50 #default=0.5

# ---------------------------------
# ---------------------------------

# Flips the image channels from RGB to BGR with the specified probability, useful for increasing robustness to incorrect channel ordering.
# Range: 0.0 - 1.0,  % chances of being applied
BGR_CHANNEL_SWAP = 0.0 #default=0.0

# ---------------------------------
# ---------------------------------

# Combines four training images into one, simulating different scene compositions and object interactions.
# Highly effective for complex scene understanding.
# Range: 0.0 - 1.0,  % chances of being applied
MOSAIC = 0.0 #default=1.0
# mosaic augmentation helps the model learn to recognize the same species across different sizes, partial occlusions, and environmental contexts


# Blends two images and their labels, creating a composite image. Enhances the model's ability to generalize by introducing label noise and visual variability.
# Range: 0.0 - 1.0,  % chances of being applied
MIXUP = 0.0 #deafault=0.0s
# Purpose: Improves model robustness and reduces overfitting. For example, in retail product recognition systems, mixup helps the model learn more robust features
# by blending images of different products, teaching it to identify items even when they're partially visible or obscured by other products on crowded store shelves.

