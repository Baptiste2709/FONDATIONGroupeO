import cv2
import cv2.aruco as aruco
import numpy as np
import os

#Taille du plateau d'echecs en nombre de cases
CHESS_BOARD_DIM = (9, 6)
#Taille d'une case en milimètres
SQUARE_SIZE = 14

#Création d'un dossier calib_data
calib_data_path = "calib_data"
CHECK_DIR = os.path.isdir(calib_data_path)
if not CHECK_DIR:
    os.makedirs(calib_data_path)
    print(f'"{calib_data_path}"  Directory is created')
else:
    print(f'"{calib_data_path}"  Directory already created')

#Création d'un dossier images
image_dir_path = "images"
CHECK_DIR = os.path.isdir(image_dir_path)
if not CHECK_DIR:
    os.makedirs(image_dir_path)
    print(f'"{image_dir_path}"  Directory is created')
else:
    print(f'"{image_dir_path}"  Directory already created')


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#Fonction de calibration de ma camera en renvoyant un fichier .npz avec les differentes valeurs de CamMatrix, distCoef, rVec et tVec
def calibration(image_dir_path, calib_data_path, gray):
    obj3D = np.zeros((CHESS_BOARD_DIM[0] * CHESS_BOARD_DIM[1], 3), np.float32)
    obj3D[:, :2] = np.mgrid[0: CHESS_BOARD_DIM[0], 0: CHESS_BOARD_DIM[1]].T.reshape(-1, 2)
    obj3D *= SQUARE_SIZE
    print(obj3D)

    obj_points_3D = []
    img_points_2D = []

    files = os.listdir(image_dir_path)
    for file in files:
        print(file)
        imagePath = os.path.join(image_dir_path, file)

        image = cv2.imread(imagePath)
        ret, corners = cv2.findChessboardCorners(image, CHESS_BOARD_DIM, None)
        if ret == True:
            obj_points_3D.append(obj3D)
            corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
            img_points_2D.append(corners2)

            img = cv2.drawChessboardCorners(image, CHESS_BOARD_DIM, corners2, ret)

    cv2.destroyAllWindows()

    ret, mtx, dist, rVec, tVec = cv2.calibrateCamera(obj_points_3D, img_points_2D, gray.shape[::-1], None, None)
    print("C'est calibré !")
    print("Importation de toutes les données")
    np.savez(f"{calib_data_path}/MultiMatrix", camMatrix=mtx, distCoef=dist, rVector=rVec, tVector=tVec)

    print("----------------------------------------------")
    print("Loading data stored using numpy savez function \n \n \n")

    data = np.load(f"{calib_data_path}/MultiMatrix.npz")
    camMatrix = data["camMatrix"]
    distCoef = data["distCoef"]
    rVector = data["rVector"]
    tVector = data["tVector"]

    print("loaded calibration data successfully")

#Fonction de détection d'un jeu d'échec pour la calibration
def detectChessBoard(image, grayImage, criteria, boardDimension):
    ret, corners = cv2.findChessboardCorners(grayImage, boardDimension)
    if ret == True:
        corners1 = cv2.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
        image = cv2.drawChessboardCorners(image, boardDimension, corners1, ret)

    return image, ret


cap = cv2.VideoCapture(0)
n = 0

#Boucle while
while True:
    _, frame = cap.read()
    copyFrame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image, board_detected = detectChessBoard(frame, gray, criteria, CHESS_BOARD_DIM)
    cv2.putText(frame, f"saved_img : {n}", (30, 40), cv2.FONT_HERSHEY_PLAIN, 1.4, (0,255,0), 2, cv2.LINE_AA)
    #Affichage de la caméra et donc sa copie
    cv2.imshow("Camera", frame)
    cv2.imshow("CameraCopy", copyFrame)

    #Commande d'arret de code
    key = cv2.waitKey(1)
    if key == ord('a'):
        break
    #Commande de sauvergarde d'une image dans le dossier Images
    if key == ord('s') and board_detected == True:
        cv2.imwrite(f"{image_dir_path}/image{n}.png", copyFrame)

        print(f"savaed image number {n}")
        n += 1

#calibration de la caméra
calibration(image_dir_path, calib_data_path, gray)

#Libération
cap.release()
cv2.destroyAllWindows()

#Affichage du nombre d'images sauvegardés
print("Total saved Images :", n)


