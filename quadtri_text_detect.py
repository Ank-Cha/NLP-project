import numpy as np
import cv2
import math
import argparse
import os
# This function extracts the vertices and also the angle of the bounding box, given the score,geometry predicted by model
def extract_features(sc, geo, thr):

    detect = []
    conf = []
    # Checking possible errors in structure of Data passed in this function and printing possible errors for debugging
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

    columns = sc.shape[2]
    rows = sc.shape[3]
    #looping through each column
    for y in range(0, columns):

        # extract the scores (probabilities) and geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scores_data = sc[0][0][y]
        x0_data = geo[0][0][y]
        x1_data = geo[0][1][y]
        x2_data = geo[0][2][y]
        x3_data = geo[0][3][y]
        angles_data = geo[0][4][y]
        #looping through each row
        for x in range(0, rows):
            score = scores_data[x]

            if score < thr:
                continue
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]
            # Calculate offset
            off_x = x * 4.0
            off_y = y * 4.0
            #getting angle data of the bounding box
            angle = angles_data[x]
            #Calculating sin and cos of the angle obtained
            cos_A = math.cos(angle)
            sin_A = math.sin(angle)

            # calculating the starting and ending
            offset = ([off_x + cos_A * x1_data[x] + sin_A * x2_data[x], off_y - sin_A * x1_data[x] + cos_A * x2_data[x]])
            p1 = (-sin_A * h + offset[0], -cos_A * h + offset[1])
            p3 = (-cos_A * w + offset[0],  sin_A * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detect.append((center, (w, h), -1*angle * 180.0 / math.pi))
            conf.append(float(score))

    # Return detections and confidences
    return [detect, conf]

# Argument parser is defined to pass arguments such as images path and threshold probability  at the time of execution

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to input image')
parser.add_argument('--thr', type=float, default=0.5, help='Confidence threshold. Default: 50%')
args = parser.parse_args()

if __name__ == "__main__":
    # Read and store arguments
    confThreshold = args.thr
    nmsThreshold = 0.4
    # Setting image resolution to 320x320 to pass it to the model
    inpWidth = 320
    inpHeight = 320
    model = "EAST_Text_Detection_Pre-trained_Model.pb"

    # Loading the pre-trained dnn text detector model
    net = cv2.dnn.readNet(model)

    # Creating a new named window
    cv2.namedWindow("EAST: An Efficient and Accurate Scene Text Detector", cv2.WINDOW_NORMAL)

    #Defining the output layers of the model
    # the first is the output probability
    # second can be used to derive the bounding box coordinates of text
    OP_Layers = []
    OP_Layers.append("feature_fusion/Conv_7/Sigmoid")
    OP_Layers.append("feature_fusion/concat_3")

    # capturing images as a frame
    cap = cv2.VideoCapture(args.input if args.input else 0)

    while cv2.waitKey(1) < 0:
        # Read frame
        hasFrame, frame = cap.read()
        if not hasFrame:
            #waiting for any keypress to continue
            cv2.waitKey()
            break

        # Get frame height and width
        height_ = frame.shape[0]
        width_ = frame.shape[1]
        # setting images height and width suitable for pre-trained EAST model
        rW = width_ / float(inpWidth)
        rH = height_ / float(inpHeight)

        # Create a 4D blob from frame.
        # Performing a subtraction operation on RGB values of each pixel as a part of pre-processing the image
        blob = cv2.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

        # Run the model
        net.setInput(blob)
        # Forwarding blob in dnn to obtain output
        output = net.forward(OP_Layers)
        # displaying time taken for prediction by model
        t, _ = net.getPerfProfile()
        label = ' time taken to detect: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())

        # Getting the scores (probability) and geometries predicted by the model and feeding it
        # to the extract_features function defined and explained above (Line 7-69 in code)
        scores = output[0]
        geometry = output[1]
        [boxes, confidences] = extract_features(scores, geometry, confThreshold)
        # Apply Non max suppression: This step would remove the weak( low probability)  bounding boxes
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)

        for i in indices:
            # get 4 corners of the rotated rect
            vertices = cv2.boxPoints(boxes[i[0]])
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
            for j in range(4):
                p1 = (vertices[j][0], vertices[j][1])
                p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
                cv2.line(frame, p1, p2, (0, 100, 255), 2, cv2.LINE_AA)
                #print(vertices)
                #print("#")
                #crp = frame[int(vertices[0][0]):int(vertices[2][0]), int(vertices[0][1]):int(vertices[2][1])]
                #cv2.imshow("jj"+str(j), crp)
                #cv2.putText(frame, "{:.3f}".format(confidences[i[0]]), (vertices[0][0], vertices[0][1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 100, 255), 1, cv2.LINE_AA)
                #print(confidences[i[0]])
                ## ATTEMPT TO CROP POLYGONAL ROI FROM IMAGE FOR BETTER PRE-PROCESSING FOR TEXT RECOGNITION (FAILED) ####
        #         rect = cv2.boundingRect(vertices)
        #         x, y, w, h = rect
        #         croped = frame[y:y + h, x:x + w].copy()
        #
        #         vertices = vertices - vertices.min(axis=0)
        #
        #         mask = np.zeros(croped.shape[:2], np.uint8)
        #         cv2.drawContours(mask, [vertices], -1, (255, 255, 255), -1, cv2.LINE_AA)
        #
        #         dst = cv2.bitwise_and(croped, croped, mask=mask)
        #
        #         bg = np.ones_like(croped, np.uint8) * 255
        #         cv2.bitwise_not(bg, bg, mask=mask)
        #         dst2 = bg + dst
        #
        #         cv2.imwrite("croped.png", croped)
        #         cv2.imwrite("mask.png", mask)
        #         cv2.imwrite("dst.png", dst)
        #         cv2.imwrite("dst2.png", dst2

            cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255))

        # Display the image
        cv2.imshow(""
                   ": An Efficient and Accurate Scene Text Detector",frame)
        cv2.imwrite("out-{}".format(args.input),frame)
        # waiting for keypress to continue
        cv2.waitKey(0)