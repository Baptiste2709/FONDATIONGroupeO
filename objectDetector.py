import cv2
import numpy as np
import time
import os




class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0/200)
        self.net.setInputMean((200, 200, 200))
        self.net.setInputSwapRB(True)

        self.readClasses()


    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, "__Background__")

        self.colorList = np.random.uniform(0, 255, (len(self.classesList), 3))
        print(self.classesList)

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if(cap.isOpened() == False):
            print("Error opening File...")
            return

        (success, image) = cap.read()

        while success:
            classLabelIDs, confidences, corners = self.net.detect(image, 0.4)

            corners = list(corners)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            cornerIDs = cv2.dnn.NMSBoxes(corners, confidences, score_threshold = 0.6, nms_threshold = 0.2)

            if len(cornerIDs) != 0:
                for i in range(0, len(cornerIDs)):
                    corner = corners[np.squeeze(cornerIDs[i])]
                    classConfidence = confidences[np.squeeze(cornerIDs[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(cornerIDs[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    displayText = "{}:{:.2f}".format(classLabel, classConfidence)

                    x, y, w, h = corner

                    ####################
                    cv2.rectangle(image, (x,y), (x+w, y+h), color = (255, 255, 255), thickness = 1)
                    cv2.putText(image, displayText, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

                    lineWidth = min(int(w * 0.3), int(h * 0.3))
                    lineThickness = 4

                    cv2.line(image, (x,y), (x + lineWidth, y), classColor, lineThickness)
                    cv2.line(image, (x,y), (x, y + lineWidth), classColor, lineThickness)

                    cv2.line(image, (x + w,y), (x + w - lineWidth, y), classColor, lineThickness)
                    cv2.line(image, (x + w,y), (x + w, y + lineWidth), classColor, lineThickness)

                    cv2.line(image, (x,y + h), (x + lineWidth, y + h), classColor, lineThickness)
                    cv2.line(image, (x,y + h), (x,y + h - lineWidth), classColor, lineThickness)

                    cv2.line(image, (x + w,y + h), (x + w - lineWidth, y + h), classColor, lineThickness)
                    cv2.line(image, (x + w,y + h), (x + w,y + h - lineWidth), classColor, lineThickness)
                    #####################

            cv2.imshow("Result", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                break

            (success, image) = cap.read()
        cv2.destroyAllWindows()



videoPath = "videos/road.mp4"
configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
classesPath = os.path.join("model_data", "coco.names")

detector = Detector(videoPath, configPath, modelPath, classesPath)
detector.onVideo()

