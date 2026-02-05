# Annotation and Bounding Box for Yolov11

labels must respect Yolo format : `*.txt` where "*" corespond to the same name as the image it is link to
inside the *txt, each line represent an object in this order : `class_index x_center y_center width height`
if there is no interesting object to labels, let the file empty. But Remember that the file `*.txt` MUST STILL EXISTS for this image

Yolo Format:

- Bounding box of the important object in the image
each line must contains: `class_index x_center y_center width height`
- Coordinates must be normalise with a value between 0 and 1
- To do so, divide pixels `x_center` and `width` of the bounding box by `lenght of the image`
- divide pixels `y_center` and `height` of the bounding box by `height of the image`

class_index are annotated by number starting from 0, 1, 2, ...

Yolo Format example:
object labelled as `"person"` are assigned as `class_index 0` (class_index are defined in your .yaml file for your model)
object labelled as `"Necktie"` are assigned as `class_index 27`

- x center of bounding box: 0.48
- y center of bounding box: 0.63
- width center of bounding box : 0.69,
- height center of bounding box : 0.71

`0 0.481719 0.634028 0.690625 0.713278`
`27 0.364844 0.795833 0.078125 0.400000`

My Example
images/fire_image0002.jpg size : 1132 width x 1072 height
Bounding Box given by SAM => `[{'x': 392, 'y': 546, 'width': 626, 'height': 521, 'label': 'fire'}]`                         

class_index x_center y_center width height
0="fire" 392/1132 546/1072 626/1132 521/1072

=> inside labels/fire_image0002.txt

`0 0.365671 0.509328 0.553003 0.486007`


