import cv2
import numpy as np
import glob
import os
from camera_config import CameraCalibrationConfig

def calibrate_camera(config=None):
    """
    Calibrate camera using chessboard images
    
    Args:
        config: Configuration object. If None, uses CameraCalibrationConfig
    
    Returns:
        tuple: (camera_matrix, dist_coeffs, repError) or (None, None, None) if failed
    """
    if config is None:
        config = CameraCalibrationConfig
    
    # Validate configuration
    if not config.validate_config():
        return None, None, None
    
    chessboard_size = config.CHESSBOARD_SIZE
    square_size = config.SQUARE_SIZE
    images_path = config.IMAGE_PATH
    
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    imagesList = glob.glob(os.path.join(images_path, config.IMAGE_PATTERN))
    print(f"Looking for images in: {images_path}")
    print(f"Using pattern: {config.IMAGE_PATTERN}")
    print(f"Found {len(imagesList)} images")
    
    if os.path.exists(images_path):
        print("Image directory exists")
        try:
            files_in_dir = os.listdir(images_path)
            print(f"Files in directory: {files_in_dir[:10]}")  # Show first 10 files
        except Exception as e:
            print(f"Error listing directory: {e}")
    else:
        print("Image directory does not exist!")

    if not imagesList:
        print(f"No images found matching pattern '{config.IMAGE_PATTERN}' in {images_path}")
        print("Please check the image path and pattern in config file.")
        return None, None, None

    print(f"Processing {len(imagesList)} images...")
    successful_detections = 0
    
    for i, fname in enumerate(imagesList):
        print(f"Processing image {i+1}/{len(imagesList)}: {os.path.basename(fname)}")
        img = cv2.imread(fname)
        if img is None:
            print(f"Could not load image {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try multiple chessboard sizes in case the config is wrong
        possible_sizes = [
            chessboard_size,  # From config
            (8, 5), (5, 8),   # Your chessboard variations
            (9, 6), (6, 9),   # Standard variations
            (7, 7), (8, 6), (6, 8),  # Other common sizes
        ]
        
        ret = False
        corners = None
        detected_size = None
        
        for test_size in possible_sizes:
            ret, corners = cv2.findChessboardCorners(gray, test_size, None)
            if ret:
                detected_size = test_size
                print(f"  ✓ Chessboard detected with size {test_size} ({successful_detections + 1} successful so far)")
                break
        
        if not ret:
            print(f"  ✗ No chessboard detected with any common size")
            continue
            
        successful_detections += 1
        
        # If detected size is different from config, we need to recalculate objp
        if detected_size != chessboard_size:
            print(f"  ! Note: Using detected size {detected_size} instead of config size {chessboard_size}")
            temp_objp = np.zeros((detected_size[0]*detected_size[1], 3), np.float32)
            temp_objp[:, :2] = np.mgrid[0:detected_size[0], 0:detected_size[1]].T.reshape(-1, 2)
            temp_objp *= square_size
            objpoints.append(temp_objp)
        else:
            objpoints.append(objp)
            
        # Use config settings for corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                   config.CORNER_CRITERIA['max_iter'], 
                   config.CORNER_CRITERIA['epsilon'])
        corners2 = cv2.cornerSubPix(gray, corners, 
                                  config.CORNER_SUBPIX_WINDOW, 
                                  config.CORNER_SUBPIX_ZERO_ZONE, 
                                  criteria)
        imgpoints.append(corners2)
        
        # Show detected corners based on config
        if config.SHOW_CORNERS:
            cv2.drawChessboardCorners(img, detected_size, corners2, ret)
            cv2.imshow('Detected Corners', img)
            cv2.waitKey(config.DISPLAY_DELAY)
        else:
            print(f"  ✗ No chessboard detected")

    cv2.destroyAllWindows()
    
    print(f"\nSummary: Successfully detected chessboard in {successful_detections}/{len(imagesList)} images")

    if not objpoints or not imgpoints:
        print("No corners were found in any image.")
        print("Please check:")
        print("- Image quality and lighting")
        print("- Chessboard size configuration (currently set to", chessboard_size, ")")
        print("- Image file format and pattern")
        return None, None, None

    print(f"Calibrating camera using {len(objpoints)} images with detected corners...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    if not ret:
        print("Camera calibration failed.")
        return None, None, None

    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)

    
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    repError = mean_error / len(objpoints)
    print("Total reprojection error:", repError)

    # Save Calibration Parameters based on config
    if config.SAVE_PARAMS:
        paramPath = os.path.join(images_path, config.OUTPUT_FILENAME)
        np.savez(paramPath, repError=repError, camMatrix=camera_matrix, distCoeff=dist_coeffs, rvecs=rvecs, tvecs=tvecs)
        print(f"Calibration parameters saved to: {paramPath}.npz")
    
    return camera_matrix, dist_coeffs, repError


if __name__ == "__main__":
    # Example usage with default config
    result = calibrate_camera()
    
    if result[0] is not None:
        camera_matrix, dist_coeffs, repError = result
        print(f"Calibration successful!")
        print(f"Reprojection error: {repError}")
    else:
        print("Calibration failed. Please check your images and configuration.")

