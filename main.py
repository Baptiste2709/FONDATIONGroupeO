import cv2
import cv2.aruco as aruco
import numpy as np
import os


#Récuperation des données stockés lors de la calibration de la caméra
calib_data_path ="calib_data/MultiMatrix.npz"
calib_data = np.load(calib_data_path)
camMat = calib_data["camMatrix"]
distCoef = calib_data["distCoef"]
rVec = calib_data["rVector"]
tVec = calib_data["tVector"]


cap = cv2.VideoCapture(0)

#Fonction permettant la localisation des marqueurs arUco
#En precisant leur distance par rapport à la caméra, leur coordonnées et leurs ids
def findArucoMarkers(frame, gray, markerSize = 5, totalMarkers = 250):
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    markersCorners, markersIds, rejected = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)

    if markersCorners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(markersCorners, markerSize, camMat, distCoef)
        total_markers = range(0, markersIds.size)

        for ids, corners, i in zip(markersIds, markersCorners, total_markers):
            aruco.drawDetectedMarkers(frame, markersCorners)
            corners = corners.reshape(4,2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()

            point = cv2.drawFrameAxes(frame, camMat, distCoef, rVec[i], tVec[i], 4, 4)

            cv2.putText(frame, f" id: {ids[0]} Dist : {round(tVec[i][0][2], 2)}", top_right, cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f" x: {round(tVec[i][0][0], 1)} y:{round(tVec[i][0][1], 1)}", bottom_left, cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), 1, cv2.LINE_AA)

    return [markersCorners, markersIds]

#Boucle while
def main():
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        findArucoMarkers(frame, gray)
        cv2.imshow("Camera", frame)


        if cv2.waitKey(1) == ord('a'):
            break

    cap.release()
    cv2.destroyAllWindows()


main()