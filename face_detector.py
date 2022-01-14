from batch_face import RetinaFace
import argparse
import numpy as np
from time import time
import cv2
import os
# Creating the parser
parser = argparse.ArgumentParser()
# Adding arguments
parser.add_argument('--input', type=str, required=True, help="path to your image or directory of images (required)")
parser.add_argument('--output', type=str, required=True, help="path to output directory (required)")

parser.add_argument('--deblur', action='store_true', help="only applies deblur if the images is recognized as blurry")
parser.add_argument('--thr', type=float, default=0.8, help="face detector threshold")
parser.add_argument('--use_gpu', action='store_true', help="use gpu for face detection")
parser.add_argument('--network', type=str, default="resnet50",
                    help="select the backbone for face detector (mobilenet/resnet50)")
parser.add_argument('--weight', type=str,
                    help="give the local path or leave it alone to download the weight automatically")
parser.add_argument('--allowed_formats', type=list, default=['jpg', 'jpeg', 'png'], help="allowed formats for images")


# Parsing the arguments
args = parser.parse_args()

# creating the face detector with given parameters
start = time()
detector = RetinaFace(gpu_id=0 if args.use_gpu else -1, network=args.network,
                      model_path=args.weight if args.weight else None)
print('Time taken to create the face detector: ', time()-start)

# checking if input is a single image or a directory
is_directory = False
if os.path.isfile(args.input):
    paths = [os.path.basename(args.input)]
    args.input = os.path.dirname(args.input)
else:
    paths = os.listdir(args.input)
    paths = [path for path in paths if path.split(".")[-1] in args.allowed_formats]
    is_directory = True

# detecting all faces in every given image and drawing the corresponding bounding box for each face
for path in paths:
    img = cv2.imread(os.path.join(args.input, path))

    if args.deblur:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.Laplacian(gray, cv2.CV_64F).var()
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        # checking the variance of laplacian to check whether an image is blurry or not
        if variance < 150:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            img = cv2.filter2D(img, -1, kernel)

    # detecting faces
    start = time()
    faces = detector(img, threshold=args.thr, cv=True)
    print(f'Time taken to detect face(s) for {path}: ', time() - start)

    # drawing bounding box(es)
    start = time()
    for box, landmarks, score in faces:
        (x, y, x2, y2) = box
        start_point = (int(x), int(y))
        end_point = (int(x2), int(y2))
        color = (0, 255, 0)
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.rectangle(img, start_point, end_point, color, thickness)
    img = cv2.putText(img, 'Total faces: '+str(len(faces)), (150, 150), fontFace=font,
                      fontScale=6, color=(255, 0, 255), thickness=10)
    print(f'Time taken to draw all bounding boxes: ', time() - start)
    cv2.imwrite(os.path.join(args.output, path.split(".")[0] + "_faces." + path.split(".")[1]), img)
    print('\n')
