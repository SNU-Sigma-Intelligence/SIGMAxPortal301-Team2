"""
Configuration file for camera calibration
Modify these settings according to your setup
"""

import os

# Camera Calibration Settings
class CameraCalibrationConfig:
    # Chessboard pattern settings
    CHESSBOARD_SIZE = (8, 5)  # (width, height) in number of inner corners
    SQUARE_SIZE = 0.025  # Size of each square in meters (25mm)
    
    # Image settings
    IMAGE_PATH = "/home/woong/SIGMAxPortal301-Team2/T1/WC/captured_images"  # Path to calibration images
    IMAGE_PATTERN = "*.jpg"  # Pattern to match calibration images
    # Alternative pattern for frame images: "frame*.jpg"
    
    # Corner detection settings
    CORNER_SUBPIX_WINDOW = (11, 11)  # Window size for corner refinement
    CORNER_SUBPIX_ZERO_ZONE = (-1, -1)  # Zero zone for corner refinement
    CORNER_CRITERIA = {
        'type': 'EPS_AND_MAX_ITER',  # Termination criteria type
        'max_iter': 30,  # Maximum number of iterations
        'epsilon': 0.001  # Required accuracy
    }
    
    # Display settings
    SHOW_CORNERS = True  # Whether to display detected corners
    DISPLAY_DELAY = 200  # Delay in milliseconds for image display
    
    # Output settings
    SAVE_PARAMS = True  # Whether to save calibration parameters
    OUTPUT_FILENAME = "camera_calibration_params"  # Output file name (without extension)
    
    # Validation settings
    CALCULATE_REPROJECTION_ERROR = True  # Whether to calculate reprojection error
    
    @classmethod
    def get_absolute_image_path(cls):
        """Get the absolute path to the images directory"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(current_dir, cls.IMAGE_PATH))
    
    @classmethod
    def validate_config(cls):
        """Validate the configuration settings"""
        image_path = cls.get_absolute_image_path()
        if not os.path.exists(image_path):
            print(f"Warning: Image path does not exist: {image_path}")
            return False
        
        if cls.SQUARE_SIZE <= 0:
            print("Error: Square size must be positive")
            return False
            
        if any(dim <= 0 for dim in cls.CHESSBOARD_SIZE):
            print("Error: Chessboard dimensions must be positive")
            return False
            
        return True

# Alternative configurations for different setups
class AlternativeConfigs:
    """Alternative configuration presets"""
    
    @staticmethod
    def get_frame_based_config():
        """Configuration for frame-based calibration images"""
        config = CameraCalibrationConfig()
        config.IMAGE_PATTERN = "frame*.jpg"
        return config
    
    @staticmethod
    def get_high_precision_config():
        """Configuration for high-precision calibration"""
        config = CameraCalibrationConfig()
        config.CORNER_CRITERIA['max_iter'] = 100
        config.CORNER_CRITERIA['epsilon'] = 0.0001
        config.CORNER_SUBPIX_WINDOW = (5, 5)
        return config
