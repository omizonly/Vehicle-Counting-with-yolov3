# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 21:11:21 2020

@author: Osman
"""

import numpy as np
import argparse
import imutils
import cv2
import os

from sort import *
tracker = Sort()
memory = {}
#change the counting line position here
line = [(0, 200), (1280, 200)]
counter = 0


ap = argparse.ArgumentParser()
ap.add_argument('--input', type= str, default='input/cars.mp4', help='path of input video')
ap.add_argument('--output', type= str, default= 'output/cars.mp4', help='path of output video')
ap.add_argument('--yolo', type= str, default= 'yolo', help='path of yolo models')
ap.add_argument('--threshold', type= float, default= 0.3, help='insert threshold value')
ap.add_argument('--confidence', type= float, default= 0.5, help='insert conf. value')
args = vars(ap.parse_args())

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

labels_path = os.path.sep.join([args["yolo"],"coco.names"])
Labels = open(labels_path).read().strip().split("\n")

np.random.seed(42)

weights_path = os.path.sep.join([args["yolo"], "yolov3.weights"])
config_path = os.path.sep.join([args["yolo"], "yolov3.cfg"])

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vid= cv2.VideoCapture(args["input"])
writer = None
(H,W)=(None, None)

frameIndex = 0

prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
    else cv2.CAP_PROP_FRAME_COUNT
total = int(vid.get(prop))
print("No. of frames in video: {}".format(total))

while True:
    (grabbed, frame) = vid.read()
    if not grabbed:
        break
    
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence
            scores = detection[5:]
            classID = np.argmax(scores)
            if classID == 2:
                confidence = scores[classID]
            else:
                continue
            
            # filter out weak predictions
            if confidence > args["confidence"]:
                # scale the bounding box coordinates
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # get the top-left coordinates of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update the list of bounding box coordinates,confidences, and class IDs
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                
                classIDs.append(classID)
                
    # apply non-max suppression 
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
    
    dets = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x+w, y+h, confidences[i]])

    #np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    c = []
    previous = memory.copy()
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    if len(boxes) > 0:
        i = int(0)
        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            # draw a bounding box rectangle and label on the image
            color = [95,179,61]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            if indexIDs[i] in previous:
                previous_box = previous[indexIDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                cv2.line(frame, p0, p1, color, 3)

                if intersect(p0, p1, line[0], line[1]):
                    counter += 1


            text = "{}".format(indexIDs[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 2)
            i += 1

    # draw line
    cv2.line(frame, line[0], line[1], (0, 255, 255), 5)

    # draw counter
    cv2.putText(frame, str(counter), (100,200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 5.0, (0, 255, 255), 10)


    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            (frame.shape[1], frame.shape[0]), True)

    # write the output frame to disk
    writer.write(frame)
    frameIndex +=1
    print("Processing Frame ID: %d"%frameIndex)

writer.release()
vid.release()