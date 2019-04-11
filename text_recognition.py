
#IMPORTING REQUIRED MODULES
import keyboard
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2
import pytesseract
import math

# Maam, You need to install tessaract and change this path accordingly
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# An argument parser is defined to pass required arguments to the program while executing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,)
ap.add_argument("-p", "--padding", type=float, default=0.05,
	help="amount of padding to add to ROI")
args = vars(ap.parse_args())

# load the input image and get the image dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(origH, origW) = image.shape[:2]

# set the new width and height and then determine the ratio in change
(newW, newH) = (320, 320)
rW = origW / float(newW)
rH = origH / float(newH)

# resize the image and get new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]


def extract_features(sc, geo):
	if len(sc.shape) != 4:
		print("Error in len(sc.shape)")

	if len(geo.shape) != 4:
		print("Error in len(geo.shape)")

	if sc.shape[0] != 1:
		print("Error in sc.shape[0]")

	if geo.shape[0] != 1:
		print("Error in geo.shape[0]")

	if sc.shape[1] != 1:
		print("Error in sc.shape[1]")

	if geo.shape[1] != 5:
		print("Error in geo.shape[1]")

	if sc.shape[2] != geo.shape[2]:
		print("Error in sc.shape[2] or geo.shape[2]")

	if sc.shape[3] != geo.shape[3]:
		print("Error in sc.shape[3] or geo.shape[3]")


	(numRows, numCols) = sc.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
			# extract the scores (probabilities), followed by the
			# geometrical data used to derive potential bounding box
			# coordinates that surround text
			scores_data = sc[0, 0, y]
			x_data0 = geo[0, 0, y]
			x_data1 = geo[0, 1, y]
			x_data2 = geo[0, 2, y]
			x_data3 = geo[0, 3, y]
			angles_data = geo[0, 4, y]

			# loop over the number of columns
			for x in range(0, numCols):
				# if our score does not have enough probability,
				# ignore it
				if scores_data[x] < args["min_confidence"]:
					continue

				(offset_x, offset_y) = (x * 4.0, y * 4.0)

				# extracting the rotation angle for the prediction and
				# computing sin and cosine values
				angle = angles_data[x]
				cos = np.cos(angle)
				sin = np.sin(angle)

				# use the geometry volume to derive the width and height
				# of the bounding box
				h = x_data0[x] + x_data2[x]
				w = x_data1[x] + x_data3[x]

				# compute both the starting and ending (x, y)-coordinates
				# for the text prediction bounding box
				endX = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
				endY = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
				startX = int(endX - w)
				startY = int(endY - h)

				# add the bounding box coordinates and probability score
				# to our respective lists
				rects.append((startX, startY, endX, endY))
				confidences.append(scores_data[x])

			# return a tuple of the bounding boxes and associated confidences
			return (rects, confidences)



# define the two output layer names for the model:
# the first is the output probability
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detection model
print("Loading EAST pre-trained model")
net = cv2.dnn.readNet("EAST_Text_Detection_Pre-trained_Model.pb")

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
#print(geometry)


# suppress weak, overlapping bounding boxes (non-maxima suppression)
(rects, confidences) = extract_features(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

results = []

for (startX, startY, endX, endY) in boxes:

	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	# computing padding
	dX = int((endX - startX) * args["padding"])
	dY = int((endY - startY) * args["padding"])

	#Computing the Start and End x,y coordinates of the bounding box
	startX = max(0, startX - dX)
	startY = max(0, startY - dY)
	endX = min(origW, endX + (dX * 2))
	endY = min(origH, endY + (dY * 2))

	# extract the actual padded ROI
	roi = orig[startY:endY, startX:endX]

	# Tesseract v4 OCR engine:
	# (1) a language, (2) an OEM flag of 4, indicating that the we
	# wish to use the LSTM neural net model for OCR, and finally
	# (3) an OEM value, in this case, 7 which implies that we are
	# treating the ROI as a single line of text
	config = ("-l eng --oem 1 --psm 7")
	text = pytesseract.image_to_string(roi, config=config)
	# Storing predicted results of recognized text and bounding box coordinates
	results.append(((startX, startY, endX, endY), text))

results = sorted(results, key=lambda r:r[0][1])

# The following code is for custom user selection for logging selective recognized results
St=""
for ((startX, startY, endX, endY), text) in results:
	# display the text OCR'd by Tesseract
	print("OCR TEXT")
	print("========")
	print("{}\n".format(text))

	# usless char strip
	text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
	output = orig.copy()
	cv2.rectangle(output, (startX, startY), (endX, endY),
		(0, 0, 255), 2)
	cv2.putText(output, text, (startX, startY - 2),
		cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
	cv2.imshow("Text Detection:", cv2.resize(output, (960, 540)))
	if(cv2.waitKey(0)==ord('a')):
	# show the output image
	#if(St==""):
		St=St+text
		#print("YYYYTTTTTT")
	while True:  # making a loop to go through all text recognitions one by one
			# Recognized text only logged if user accepts the highlighted box and recognized text
			# User should press 'a' to accept and any other key to reject
			if keyboard.is_pressed('a'):
				St = St + " " + text
				print("success")
				break  # finishing the loop
			else:
				break
# Log recognized text in a text file ( used for Buy the product mode)
f = open("data.txt", "w")
f.write(St)
f.close()
print(St)
