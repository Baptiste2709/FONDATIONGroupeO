import cv2
import numpy as np

#BLUE
lower_blue = np.array([100, 100, 20])
upper_blue = np.array([140, 255, 255])
#GREEN
lower_green = np.array([45, 100, 20])
upper_green = np.array([80, 255, 255])
#RED
lower_red = np.array([0, 125, 20])
upper_red = np.array([8, 255, 255])
#YELLOW
lower_yellow = np.array([25, 100, 20])
upper_yellow = np.array([35, 255, 255])

cap = cv2.VideoCapture(0)

#Boucle while
while True:
    success, frame = cap.read()
    #Passage d'un code couleur Blue_Green_Red à Hue_Saturation_Value
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #Creation des différents masques
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    #Recherche des contours des différentes couleurs
    blue_contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Creation et affichage des rectangles pour chaque couleurs
    if len(blue_contours) != 0:
        for contour in blue_contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x,y), (x + w, y + w), (255, 0, 0), 2)
    if len(green_contours) != 0:
        for contour in green_contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x,y), (x + w, y + w), (0, 255, 0), 2)
    if len(red_contours) != 0:
        for contour in red_contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x,y), (x + w, y + w), (0, 0, 255), 2)
    if len(yellow_contours) != 0:
        for contour in yellow_contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x,y), (x + w, y + w), (0, 255, 255), 2)

    #Affichage de la caméra avec les rectangles et des différents masques
    cv2.imshow("Camera", frame)
    cv2.imshow("blueMask", blue_mask)
    cv2.imshow("greenMask", green_mask)
    cv2.imshow("redMask", red_mask)
    cv2.imshow("yellowMask", yellow_mask)
    #Commande d'arret de code
    if cv2.waitKey(1) == ord('a'):
        break

#Libération
cap.release()
cv2.destroyAllWindows()