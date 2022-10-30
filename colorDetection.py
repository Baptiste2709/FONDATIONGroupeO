import cv2
import numpy as np

#BLUE
lower_blue = np.array([90, 100, 40])
upper_blue = np.array([107, 255, 255])
#DARK_BLUE
lower_d_blue = np.array([115, 50, 10])
upper_d_blue = np.array([125, 255, 255])
#GREEN
lower_green = np.array([45, 100, 15])
upper_green = np.array([80, 255, 255])
#RED
lower_red = np.array([170, 100, 20])
upper_red = np.array([180, 255, 255])
#FUSHIA
lower_fushia = np.array([150, 100, 20])
upper_fushia = np.array([163, 255, 255])
#YELLOW
lower_yellow = np.array([20, 100, 20])
upper_yellow = np.array([35, 255, 255])
#DARK
lower_dark = np.array([0, 0, 0])
upper_dark = np.array([255, 255, 30])
#WHITE
lower_white = np.array([0, 0, 0])
upper_white = np.array([255, 15, 255])

cap = cv2.VideoCapture(0)


#Boucle while
while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (500, 300), cv2.INTER_AREA)
    #Passage d'un code couleur Blue_Green_Red à Hue_Saturation_Value
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #Creation des différents masques
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue = cv2.bitwise_and(frame, frame, mask = blue_mask)
    d_blue_mask = cv2.inRange(hsv, lower_d_blue, upper_d_blue)
    d_blue = cv2.bitwise_and(frame, frame, mask = d_blue_mask)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green = cv2.bitwise_and(frame, frame, mask = green_mask)
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    red = cv2.bitwise_and(frame, frame, mask = red_mask)
    fushia_mask = cv2.inRange(hsv, lower_fushia, upper_fushia)
    fushia = cv2.bitwise_and(frame, frame, mask = fushia_mask)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow = cv2.bitwise_and(frame, frame, mask = yellow_mask)
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    dark = cv2.bitwise_and(frame, frame, mask = dark_mask)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white = cv2.bitwise_and(frame, frame, mask = white_mask)
    #Recherche des contours des différentes couleurs
    blue_contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    d_blue_contours, hierarchy = cv2.findContours(d_blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fushia_contours, hierarchy = cv2.findContours(fushia_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dark_contours, hierarchy = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white_contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Creation et affichage des rectangles pour chaque couleurs
    if len(blue_contours) != 0:
        for contour in blue_contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x,y), (x + w, y + w), (255, 190, 0), 2)
    if len(d_blue_contours) != 0:
        for contour in d_blue_contours:
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
    if len(fushia_contours) != 0:
        for contour in fushia_contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x,y), (x + w, y + w), (255, 0, 232), 2)
    if len(yellow_contours) != 0:
        for contour in yellow_contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x,y), (x + w, y + w), (0, 255, 255), 2)
    if len(dark_contours) != 0:
        for contour in dark_contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x,y), (x + w, y + w), (0, 0, 0), 2)
    if len(white_contours) != 0:
        for contour in white_contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x,y), (x + w, y + w), (255, 255, 255), 2)

    #Affichage de la caméra avec les rectangles et des différents masques
    cv2.imshow("Camera", frame)
    cv2.imshow("blue", blue)
    cv2.imshow("dBlue", d_blue)
    cv2.imshow("dark", dark)
    cv2.imshow("green", green)
    cv2.imshow("red", red)
    cv2.imshow("fushia", fushia)
    cv2.imshow("yellow", yellow)
    cv2.imshow("white", white)
    #Commande d'arret de code
    if cv2.waitKey(1) == ord('a'):
        break

#Libération
cap.release()
cv2.destroyAllWindows()