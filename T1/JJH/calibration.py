import cv2
import numpy as np
import glob

def calibrate_camera(checkerboard_images_path, checkerboard_size=(9, 6), square_size=1.0):
    """
    Camera calibration function using checkerboard images.
    
    Args:
    - checkerboard_images_path: Path to the folder containing checkerboard images.
    - checkerboard_size: Tuple of (number_of_columns, number_of_rows) of the checkerboard.
    - square_size: Size of the square in real world units (e.g., in cm or mm).
    
    Returns:
    - camera_matrix: The intrinsic camera matrix.
    - dist_coeffs: The distortion coefficients.
    """
    # Prepare object points (3D points) based on the checkerboard size and square size
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.indices(checkerboard_size).T.reshape(-1, 2)
    objp *= square_size  # Scale the object points by the square size
    
    # Arrays to store object points and image points from all the images
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane
    
    # Get all images from the specified folder
    images = glob.glob(checkerboard_images_path + "/*.jpg")
    
    for img_path in images:
        # Read the image
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret:
            obj_points.append(objp)
            img_points.append(corners)
            
            # Draw the corners on the image (optional, for visualization)
            cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
            cv2.imshow('Checkerboard', img)
            cv2.waitKey(500)  # Display image for 500ms
        else:
            print(f"Checkerboard not found in {img_path}")
    
    cv2.destroyAllWindows()
    
    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    
    if not ret:
        print("Calibration failed!")
        return None, None
    
    print("Camera calibration successful!")
    print(f"Camera Matrix:\n{camera_matrix}")
    print(f"Distortion Coefficients:\n{dist_coeffs}")
    
    return camera_matrix, dist_coeffs