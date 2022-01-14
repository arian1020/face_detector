## General Info
In this repo, an easy-to-use python code is provided for high accuracy face detection. [batch_face](https://pypi.org/project/batch-face/) is used as the face detector which is a Pytorch implementation of [RetinaFace](https://github.com/deepinsight/insightface/tree/master/detection/retinaface).


## Setup
To run this project, install the required libraries as follows:

```
$ create a virtual env (python 3.7 or higher)
$ activate your virtual env
$ pip install -r requirements.txt
```

## How to use
By running python face_detector.py -h you can access the help description for each required or optional argument.
```
python face_detector.py -h
  --input INPUT         path to your image or directory of images (required)
  --output OUTPUT       path to output directory (required)
  --deblur DEBLUR       only applies deblur if the images is recognized as
                        blurry
  --thr THR             face detector threshold
  --use_gpu USE_GPU     use gpu for face detection
  --network NETWORK     select the backbone for face detector
                        (mobilenet/resnet50)
  --weight WEIGHT       give the local path or leave it alone to download the
                        weight automatically
  --allowed_formats ALLOWED_FORMATS
                        allowed formats for images
```

## Prediction on a single image
````
python face_detector.py --input path_to_you_image --output path_to_output_dir
````

##  Models
You can download the pretrained models here [mobilenet](https://github.com/elliottzheng/face-detection/releases/download/0.0.1/mobilenet0.25_Final.pth) and [resnet50](https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5)
