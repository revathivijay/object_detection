import urllib.request
import cv2
import numpy as np
import time

#importing the yolo algorithm using weights and config file
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

#importing class names from coco.names file
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes), 3))

# Code for connecting mobile feed to PC using URL
URL = "http://192.168.10.6:8080/shot.jpg"
while True:
    img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()),dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    height, width, channels = img.shape
    # print(height," ",  width)
    #detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence>0.5:
                centre_x = int(detection[0]*width)
                centre_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(centre_x-w/2)
                y = int(centre_y-h/2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            # print(label)
            confidence_label = confidences[i]
            new_label = "(" + str(confidence_label) + ")"
            label = label + new_label
            # print(label)
            color = colors[i]
            cv2.rectangle(img, (x,y), (x*w, y*h), color, 2)
            cv2.putText(img, label, (x,y), font, 3, color, 3)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600,600)
    cv2.imshow('image', img)

    key = cv2.waitKey(1)
    if key==27:
        break
