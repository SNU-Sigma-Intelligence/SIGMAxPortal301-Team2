import cv2
import numpy as np
import glob
from cameraCalibration import *
from poseEstimation import *
import sys
import os
import importlib.util

src_dir = r"C:\Users\tad1i\projects\SIGMAxPortal301-Team2\src"

sys.path.append(src_dir)

src_path = r"C:\Users\tad1i\projects\SIGMAxPortal301-Team2\src\wrapper.py"
spec = importlib.util.spec_from_file_location("wrapper", src_path)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)




videoPath = r"C:\Users\tad1i\projects\SIGMAxPortal301-Team2\T1\TAD\chessboard pictures\long_145150.mp4"
imagesPath = r"C:\Users\tad1i\projects\SIGMAxPortal301-Team2\T1\TAD\chessboard pictures\extFrames"
cameraMatrixPath = r"C:\Users\tad1i\projects\SIGMAxPortal301-Team2\T1\TAD\chessboard pictures"
wrapper.extract_frames(videoPath, imagesPath, 1)


print("Exists:", os.path.exists(imagesPath))
print("Files:", os.listdir())

area = 4
# calibrate_camera(imagesPath, area)

perImageRmse, all_errors = reprError(cameraMatrixPath, area, imagesPath)
computeReprError(perImageRmse, all_errors)



