import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from os import getcwd

def calibrateCamera(showPics=True):
    # Load images
    root = getcwd()
    calibrationInput = os.path.join(root, 'demoImages/calibration')
    imgPathList = glob.glob(os.path.join(calibrationInput, '*.jpg'))

    # Initialize parameters
    nRows = 9
    nCols = 6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    worldPtsList = []
    imgPtsList = []

    # Find Corners
    for imgPath in imgPathList:
        imgBGR = cv2.imread(imgPath)
        imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
        
        cornersFound, corners = cv2.findChessboardCorners(imgGray, (nRows, nCols), None)
        
        if cornersFound == True:
            worldPtsList.append(worldPtsList)
            cornersRefined = cv2.cornerSubPix(imgGray, corners, (11, 11), (-1, -1), criteria)
            imgPtsList.append(cornersRefined)
            
            if showPics:
                cv2.drawChessboardCorners(imgBGR, (nRows, nCols), cornersRefined, cornersFound)
                cv2.imshow("Chessboard", imgBGR)
                cv2.waitKey(500)
    cv2.destroyAllWindows()

    # Calibrate
    camMatrix, distCoeff, rvecs, tvecs = cv2.calibrateCamera(worldPtsList, imgPtsList, imgGray.shape[::-1], None, None)
    print("Camera Matrix:", camMatrix)
    print("Reprojection Error (pixels): {:.4f}".format(reprojError))

    # Save Calibration Parameters (later video)
    paramFolder = os.path.dirname(os.path.abspath(__file__))
    paramFile = os.path.join(paramFolder, 'calibration.npy')
    np.save(paramFile, camMatrix)

    return camMatrix, distCoeff, rvecs, tvecs


def removeDistortion(camMatrix, distCoeff):
    # Remove distortion
    root = getcwd()
    imgPath = os.path.join(root, 'demoImages/distortion')
    img = cv2.imread(imgPath)
    
    h, w = img.shape[:2]
    newCamMatrix, roi = cv2.getOptimalNewCameraMatrix(camMatrix, distCoeff, (w, h), 1, (w, h))
    
    imgUndist = cv2.undistort(img, camMatrix, distCoeff, None, newCamMatrix)
    
    cv2.imwrite('undistorted_image.jpg', imgUndist)
    return imgUndist


# Draw Lines to See Distortion Change
img = cv2.imread(imgPath)
cv2.line(img, (1769,1033), (1780,922), (255,255,255), 2)
cv2.line(img, (1769,1033), (1780,922), (255,255,255), 2)

plt.figure()
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(imgUndist)
plt.show()

# Run Calibration
def runCalibration():
    calibrateCamera(showPics=True)
    runRemoveDistortion()
    return

if __name__ == '__main__':
    runCalibration()
