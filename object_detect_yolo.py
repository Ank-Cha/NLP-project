
# importing required modules
import keyboard
import numpy as np
import argparse
import time
import cv2
import os

# An argument parser is defined to pass required arguments to the program while executing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# loading the COCO dataset labels on which YOLO model was trained to detect:
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initializing a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# importing the weights and configuration of YOLO v3 pre-trained model
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

print("[INFO] loading YOLO...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# loading the input image and get its width and height
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# getting output layer names from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

boxes = []
confidences = []
classIDs = []

# Looping through the outputs of all output layers of YOLO v3
for output in layerOutputs:

	#looping through all detections of each output
	for detection in output:

		#assigning required values to intiutive variable names
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# Rejecting cases where score(probability is less than the minimum confidence defined
		if confidence > args["confidence"]:

			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")
			# use the center (x, y)-coordinates to derive the top and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# Non-maxima supression
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

# The following code is for custom user selection for logging selective recognized results
St=" "

if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}".format(LABELS[classIDs[i]])
		conf = "{:.4f}".format(confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 1)
		cv2.imshow("Object Detection:", cv2.resize(image, (960, 540)))
		if (cv2.waitKey(0) == ord('a')):
			# show the output image
			# if(St==""):
			St = St + text

		print(text)
		while True:   # making a loop to go through all text recognitions one by one
			# Recognized text only logged if user accepts the highlighted box and recognized text
			# User should press 'a' to accept and any other key to reject
			if keyboard.is_pressed('a'):
				St = St + " " + text
				print("success")
				break  # finishing the loop
			else:
				break
# logging detected object information in data.txt ( used in buy the product mode)
f = open("data.txt", "a")
f.write(St)
print(St)
f.close()

#cv2.imshow("Image", cv2.resize(image, (960, 540)))
cv2.waitKey(0)