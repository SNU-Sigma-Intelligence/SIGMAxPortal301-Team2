"""
Configuration file for pose estimation
Modify these settings according to your setup
"""

import os
import cv2

class PoseEstimationConfig:
    # File paths
    CALIBRATION_FILE = "/home/woong/SIGMAxPortal301-Team2/T1/WC/captured_images/camera_calibration_params.npz"
    IMAGES_DIRECTORY = "/home/woong/SIGMAxPortal301-Team2/T1/WC/marker_images"
    OUTPUT_CSV_FILE = "pose_estimation_results.csv"
    
    # ArUco marker settings
    MARKER_SIZE_METERS = 0.05  # Size of ArUco markers in meters (5cm)
    ARUCO_DICTIONARY = cv2.aruco.DICT_6X6_250  # ArUco dictionary type
    
    # Image processing settings
    SUPPORTED_IMAGE_FORMATS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    USE_CLAHE = True  # Use Contrast Limited Adaptive Histogram Equalization
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_GRID_SIZE = (8, 8)
    
    # Pose estimation settings
    SOLVEPNP_METHOD = cv2.SOLVEPNP_IPPE  # PnP solving method
    # Alternative methods: cv2.SOLVEPNP_P3P, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE

    # Visualization settings
    SHOW_VISUALIZATION = True  # Whether to show pose visualization
    VISUALIZATION_DELAY = 100  # Delay in milliseconds for visualization
    DRAW_COORDINATE_AXES = True  # Draw 3D coordinate axes
    AXIS_LENGTH_RATIO = 1.0  # Axis length as ratio of marker size
    
    # Display colors (BGR format)
    COLOR_DETECTED_CORNERS = (0, 255, 0)  # Green
    COLOR_PROJECTED_CORNERS = (0, 0, 255)  # Red
    COLOR_CONNECTION_LINES = (255, 255, 0)  # Cyan
    COLOR_X_AXIS = (0, 0, 255)  # Red
    COLOR_Y_AXIS = (0, 255, 0)  # Green
    COLOR_Z_AXIS = (255, 0, 0)  # Blue
    COLOR_TEXT = (255, 255, 255)  # White
    
    # Analysis settings
    QUALITY_THRESHOLDS = {
        'excellent': 0.5,  # pixels
        'good': 1.0,       # pixels
        'acceptable': 2.0  # pixels
    }
    
    # Output settings
    SAVE_DETAILED_RESULTS = True  # Save detailed CSV results
    INCLUDE_ROTATION_DEGREES = True  # Include rotation in degrees (not just radians)
    PRINT_PROGRESS = True  # Print processing progress
    PRINT_INDIVIDUAL_RESULTS = True  # Print results for each marker
    
    @classmethod
    def get_absolute_calibration_path(cls):
        """Get the absolute path to the calibration file"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(current_dir, cls.CALIBRATION_FILE))
    
    @classmethod
    def get_absolute_images_path(cls):
        """Get the absolute path to the images directory"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(current_dir, cls.IMAGES_DIRECTORY))
    
    @classmethod
    def get_absolute_output_path(cls):
        """Get the absolute path for the output CSV file"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(current_dir, cls.OUTPUT_CSV_FILE))
    
    @classmethod
    def validate_config(cls):
        """Validate the configuration settings"""
        issues = []
        
        # Check calibration file
        calib_path = cls.get_absolute_calibration_path()
        if not os.path.exists(calib_path):
            issues.append(f"Calibration file not found: {calib_path}")
        
        # Check images directory
        images_path = cls.get_absolute_images_path()
        if not os.path.exists(images_path):
            issues.append(f"Images directory not found: {images_path}")
        
        # Check marker size
        if cls.MARKER_SIZE_METERS <= 0:
            issues.append("Marker size must be positive")
        
        # Check quality thresholds
        thresholds = cls.QUALITY_THRESHOLDS
        if not (thresholds['excellent'] < thresholds['good'] < thresholds['acceptable']):
            issues.append("Quality thresholds must be in ascending order")
        
        if issues:
            print("Configuration validation issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        return True
    
    @classmethod
    def print_config_summary(cls):
        """Print a summary of current configuration"""
        print("=" * 60)
        print("POSE ESTIMATION CONFIGURATION")
        print("=" * 60)
        print(f"Calibration file:    {cls.get_absolute_calibration_path()}")
        print(f"Images directory:    {cls.get_absolute_images_path()}")
        print(f"Output file:         {cls.get_absolute_output_path()}")
        print(f"Marker size:         {cls.MARKER_SIZE_METERS}m")
        print(f"ArUco dictionary:    {cls.ARUCO_DICTIONARY}")
        print(f"PnP method:          {cls.SOLVEPNP_METHOD}")
        print(f"Visualization:       {'Enabled' if cls.SHOW_VISUALIZATION else 'Disabled'}")
        print(f"CLAHE enhancement:   {'Enabled' if cls.USE_CLAHE else 'Disabled'}")
        print()

# Alternative configurations for different setups
class PoseEstimationPresets:
    """Predefined configuration presets"""
    
    @staticmethod
    def get_high_precision_config():
        """Configuration for high-precision pose estimation"""
        config = PoseEstimationConfig()
        config.SOLVEPNP_METHOD = cv2.SOLVEPNP_IPPE
        config.QUALITY_THRESHOLDS = {
            'excellent': 0.25,
            'good': 0.5,
            'acceptable': 1.0
        }
        config.VISUALIZATION_DELAY = 500  # Slower for careful inspection
        return config
    
    @staticmethod
    def get_fast_processing_config():
        """Configuration for fast processing (less visualization)"""
        config = PoseEstimationConfig()
        config.SHOW_VISUALIZATION = False
        config.PRINT_INDIVIDUAL_RESULTS = False
        config.DRAW_COORDINATE_AXES = False
        return config
    
    @staticmethod
    def get_debug_config():
        """Configuration for debugging with detailed output"""
        config = PoseEstimationConfig()
        config.VISUALIZATION_DELAY = 2000  # 2 seconds per image
        config.PRINT_PROGRESS = True
        config.PRINT_INDIVIDUAL_RESULTS = True
        config.SAVE_DETAILED_RESULTS = True
        return config
    
    @staticmethod
    def get_large_marker_config():
        """Configuration for large markers (10cm)"""
        config = PoseEstimationConfig()
        config.MARKER_SIZE_METERS = 0.10  # 10cm markers
        config.AXIS_LENGTH_RATIO = 0.8  # Slightly smaller axes
        return config
