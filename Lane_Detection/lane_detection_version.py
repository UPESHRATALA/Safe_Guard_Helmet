import numpy as np
import cv2
from utils import *
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_cfg', type=str, default='C:/Users/ACER/Documents/Lane-Detection/Lane_Detection/yolov3.cfg',
                    help='Path to config file')
parser.add_argument('--model_weights', type=str,
                    default='C:/Users/ACER/Documents/Lane-Detection/Lane_Detection/yolov3.weights',
                    help='Path to weights of model')
parser.add_argument('--video', type=str, default='C:/Users/ACER/Documents/Lane-Detection/Lane_Detection/sample.mp4',
                    help='Path to video file')
parser.add_argument('--src', type=int, default=0,
                    help='Source of the camera')
parser.add_argument('--output_dir', type=str, default='C:/Users/ACER/Documents/Lane-Detection/Lane_Detection',
                    help='Path to the output directory')
args = parser.parse_args()

# Print the arguments
print('----- info -----')
print('[i] The config file: ', args.model_cfg)
print('[i] The weights of model file: ', args.model_weights)
print('[i] Path to video file: ', args.video)
print('###########################################################\n')

# Verify file paths
assert os.path.isfile(args.model_cfg), f"Config file not found at {args.model_cfg}"
assert os.path.isfile(args.model_weights), f"Weights file not found at {args.model_weights}"

frameWidth = 640
frameHeight = 480

# Load the network
net = cv2.dnn.readNet(args.model_weights, args.model_cfg)
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get layer names
layers_names = net.getLayerNames()
try:
    output_layers = [layers_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except AttributeError:
    output_layers = [layers_names[i - 1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

font = cv2.FONT_HERSHEY_PLAIN
frame_id = 0
cameraFeed = False

if cameraFeed:
    intialTracbarVals = [24, 55, 12, 100]  # wT, hT, wB, hB
else:
    intialTracbarVals = [42, 63, 14, 87]  # wT, hT, wB, hB

output_file = ''
if cameraFeed:
    cap = cv2.VideoCapture(args.src)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
else:
    cap = cv2.VideoCapture(0)
    output_file = args.video[:-4].rsplit('/')[-1] + '_Detection.avi'

initializeTrackbars(intialTracbarVals)

video_writer = cv2.VideoWriter('output2.avi', cv2.VideoWriter_fourcc(*'XVID'),
                               cap.get(cv2.CAP_PROP_FPS), (2 * frameWidth, frameHeight))

starting_time = time.time()
while True:
    success, img = cap.read()
    if not success:
        print('[i] ==> Done processing!!!')
        print('[i] ==> Output file is stored at', os.path.join(args.output_dir, output_file))
        cv2.waitKey(1000)
        break

    if not cameraFeed:
        img = cv2.resize(img, (frameWidth, frameHeight), None)
    imgWarpPoints = img.copy()
    imgFinal = img.copy()
    imgCanny = img.copy()

    imgUndis = undistort(img)
    imgThres, imgCanny, imgColor = thresholding(imgUndis)
    src = valTrackbars()
    imgWarp = perspective_warp(imgThres, dst_size=(frameWidth, frameHeight), src=src)
    imgWarpPoints = drawPoints(imgWarpPoints, src)
    imgSliding, curves, lanes, ploty = sliding_window(imgWarp, draw_windows=True)

    try:
        curverad = get_curve(imgFinal, curves[0], curves[1])
        lane_curve = np.mean([curverad[0], curverad[1]])
        imgFinal = draw_lanes(img, curves[0], curves[1], frameWidth, frameHeight, src=src)

        currentCurve = lane_curve // 50
        if int(np.sum(arrayCurve)) == 0:
            averageCurve = currentCurve
        else:
            averageCurve = np.sum(arrayCurve) // arrayCurve.shape[0]
        if abs(averageCurve - currentCurve) > 200:
            arrayCurve[arrayCounter] = averageCurve
        else:
            arrayCurve[arrayCounter] = currentCurve
        arrayCounter += 1
        if arrayCounter >= noOfArrayValues:
            arrayCounter = 0
        cv2.putText(imgFinal, str(int(averageCurve)), (frameWidth // 2 - 70, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 1.75, (0, 0, 255), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"Exception: {e}")
        lane_curve = 0

    imgFinal = drawLines(imgFinal, lane_curve)

    success, frame = cap.read()
    frame = cv2.resize(frame, (frameWidth, frameHeight), None)
    frame_id += 1
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "{}: {:.2f}%".format(classes[class_ids[i]], confidences[i] * 100)
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 10), font, 2, color, 2)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS:" + str(fps), (10, 30), font, 2, (0, 0, 0), 1)

    imgBlank = np.zeros_like(img)
    imgStacked = stackImages(0.7, ([imgUndis, frame],
                                   [imgColor, imgCanny],
                                   [imgWarp, imgSliding]))

    cv2.imshow("Image", frame)
    cv2.imshow("PipeLine", imgStacked)
    cv2.imshow("Result", imgFinal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('==> All done!')
print('***********************************************************')
