import cv2
import numpy as np
import glob
import os

def calibrate_camera(imagesPath, square_size):
    chessboard_size = (9, 6)
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    imagesList = glob.glob(os.path.join(imagesPath, 'calibration*.jpg'))
    if not imagesList:
        print(f"No images found in {imagesPath}")
        return

    for fname in imagesList:
        img = cv2.imread(fname)
        if img is None:
            print(f"Could not load image {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(200)

    cv2.destroyAllWindows()

    if not objpoints or not imgpoints:
        print("No corners were found in any image.")
        return

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("Total reprojection error:", mean_error / len(objpoints))
