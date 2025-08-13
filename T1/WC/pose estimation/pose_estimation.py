import cv2
import os
import glob
import numpy as np
import csv
from datetime import datetime
from pose_config import PoseEstimationConfig, PoseEstimationPresets

def load_camera_calibration(calibration_path):
    """
    Load camera calibration parameters from npz file
    
    Args:
        calibration_path (str): Path to the calibration npz file
    
    Returns:
        tuple: (camera_matrix, dist_coeffs) or (None, None) if failed
    """
    try:
        if not os.path.exists(calibration_path):
            print(f"[ERROR] Calibration file not found: {calibration_path}")
            return None, None
            
        data = np.load(calibration_path)
        camera_matrix = data['camMatrix']
        dist_coeffs = data['distCoeff']
        
        print(f"[INFO] Loaded calibration from: {calibration_path}")
        print(f"Camera matrix:\n{camera_matrix}")
        print(f"Distortion coefficients: {dist_coeffs.flatten()}")
        
        return camera_matrix, dist_coeffs
        
    except Exception as e:
        print(f"[ERROR] Failed to load calibration: {e}")
        return None, None

def estimate_pose_from_markers(config=None):
    """
    Estimate pose from ArUco markers in images using configuration
    
    Args:
        config: Configuration object. If None, uses PoseEstimationConfig
    
    Returns:
        tuple: (per_image_results, all_errors)
    """
    if config is None:
        config = PoseEstimationConfig
    
    # Validate configuration
    if not config.validate_config():
        return [], []
    
    # Get paths from config
    calibration_path = config.get_absolute_calibration_path()
    images_path = config.get_absolute_images_path()
    
    if config.PRINT_PROGRESS:
        config.print_config_summary()
    
    # Load camera calibration
    camera_matrix, dist_coeffs = load_camera_calibration(calibration_path)
    if camera_matrix is None:
        return [], []

    # ArUco detector setup
    dictionary = cv2.aruco.getPredefinedDictionary(config.ARUCO_DICTIONARY)
    detector = cv2.aruco.ArucoDetector(dictionary)
    
    # 3D points of ArUco marker corners (assuming marker lies on XY plane)
    marker_3d_points = np.array([
        [-config.MARKER_SIZE_METERS/2,  config.MARKER_SIZE_METERS/2, 0],
        [ config.MARKER_SIZE_METERS/2,  config.MARKER_SIZE_METERS/2, 0],
        [ config.MARKER_SIZE_METERS/2, -config.MARKER_SIZE_METERS/2, 0],
        [-config.MARKER_SIZE_METERS/2, -config.MARKER_SIZE_METERS/2, 0]], dtype=np.float32)

    # Find all image files using config patterns
    image_paths = []
    for pattern in config.SUPPORTED_IMAGE_FORMATS:
        image_paths.extend(glob.glob(os.path.join(images_path, pattern)))
    
    if not image_paths:
        print(f"[ERROR] No images found in {images_path}")
        return [], []
    
    if config.PRINT_PROGRESS:
        print(f"[INFO] Found {len(image_paths)} images to process")
    
    per_image_results = []
    all_errors = []
    successful_poses = 0

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            if config.PRINT_PROGRESS:
                print(f"[ERROR] Could not read {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE if enabled in config
        if config.USE_CLAHE:
            clahe = cv2.createCLAHE(
                clipLimit=config.CLAHE_CLIP_LIMIT, 
                tileGridSize=config.CLAHE_TILE_GRID_SIZE
            )
            gray = clahe.apply(gray)

        # Detect ArUco markers
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is None:
            if config.PRINT_PROGRESS:
                print(f"[WARN] No markers found in {os.path.basename(img_path)}")
            continue

        if config.PRINT_PROGRESS:
            print(f"[INFO] Processing {os.path.basename(img_path)} - Found {len(ids)} markers")
        
        for i, marker_id in enumerate(ids.flatten()):
            marker_corners = corners[i].reshape(-1, 2)
            
            # Solve PnP to get pose using config method
            success, rvec, tvec = cv2.solvePnP(
                marker_3d_points, 
                marker_corners, 
                camera_matrix, 
                dist_coeffs, 
                flags=config.SOLVEPNP_METHOD
            )
            
            if not success:
                if config.PRINT_PROGRESS:
                    print(f"[WARN] Pose estimation failed for marker {marker_id} in {os.path.basename(img_path)}")
                continue

            # Calculate reprojection error
            projected_points, _ = cv2.projectPoints(
                marker_3d_points, rvec, tvec, camera_matrix, dist_coeffs
            )
            projected_points = projected_points.reshape(-1, 2)
            
            # Calculate errors
            errors = np.linalg.norm(projected_points - marker_corners, axis=1)
            rmse = np.sqrt(np.mean(errors**2))
            mean_error = errors.mean()
            
            # Store results
            per_image_results.append({
                'filename': os.path.basename(img_path),
                'marker_id': marker_id,
                'rmse': rmse,
                'mean_error': mean_error,
                'rvec': rvec,
                'tvec': tvec,
                'corners': marker_corners,
                'projected': projected_points
            })
            
            all_errors.extend(errors)
            successful_poses += 1

            # Visualize if enabled in config
            if config.SHOW_VISUALIZATION:
                try:
                    visualize_pose_estimation(img, marker_corners, projected_points, rvec, tvec, 
                                            camera_matrix, dist_coeffs, config)
                except Exception as e:
                    print(f"[WARN] Visualization failed for marker {marker_id}: {e}")
                    # Continue without visualization

    cv2.destroyAllWindows()
    if config.PRINT_PROGRESS:
        print(f"[INFO] Successfully estimated pose for {successful_poses} markers")
    return per_image_results, all_errors

def analyze_pose_estimation_results(per_image_results, all_errors, config):
    """
    Analyze and display comprehensive results of pose estimation
    
    Args:
        per_image_results (list): List of dictionaries containing per-image results
        all_errors (list): List of all reprojection errors
        config: PoseEstimationConfig object containing settings
    """
    if not per_image_results:
        if config.PRINT_PROGRESS:
            print("[WARN] No valid poses computed.")
        return

    print("\n" + "="*80)
    print("POSE ESTIMATION ANALYSIS RESULTS")
    print("="*80)
    
    # Convert to numpy for easier computation
    all_errors = np.array(all_errors)
    
    # Overall statistics
    global_rmse = np.sqrt(np.mean(all_errors**2))
    global_mean = all_errors.mean()
    global_std = all_errors.std()
    global_median = np.median(all_errors)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total markers processed: {len(per_image_results)}")
    print(f"  Global RMSE:            {global_rmse:.3f} pixels")
    print(f"  Global Mean Error:      {global_mean:.3f} pixels")
    print(f"  Global Std Deviation:   {global_std:.3f} pixels")
    print(f"  Global Median Error:    {global_median:.3f} pixels")
    print(f"  Min Error:              {all_errors.min():.3f} pixels")
    print(f"  Max Error:              {all_errors.max():.3f} pixels")
    
    # Per-image results
    print(f"\nPER-IMAGE RESULTS:")
    print(f"{'Filename':<25} {'Marker ID':<10} {'RMSE':<8} {'Mean Err':<10} {'Position (m)':<25}")
    print("-" * 80)
    
    for result in per_image_results:
        position = f"({result['tvec'][0][0]:.2f}, {result['tvec'][1][0]:.2f}, {result['tvec'][2][0]:.2f})"
        print(f"{result['filename']:<25} {result['marker_id']:<10} {result['rmse']:<8.3f} {result['mean_error']:<10.3f} {position:<25}")
    
    # Save results using config path
    save_results_to_csv(config.get_absolute_output_path(), 
                       per_image_results, global_rmse, global_mean)

def visualize_pose_estimation(img, corners, projected, rvec, tvec, camera_matrix, dist_coeffs, config):
    """
    Visualize pose estimation results with 3D axis and reprojection
    
    Args:
        img: Input image
        corners: Detected marker corners
        projected: Projected marker corners
        rvec, tvec: Rotation and translation vectors
        camera_matrix, dist_coeffs: Camera parameters
        config: PoseEstimationConfig object containing visualization settings
    """
    img_vis = img.copy()
    
    # Convert corners and projected points to proper format
    corners = np.array(corners, dtype=np.float32).reshape(-1, 2)
    projected = np.array(projected, dtype=np.float32).reshape(-1, 2)
    # Convert arrays to ensure proper data types
    corners = np.array(corners, dtype=np.float32).reshape(-1, 2)
    projected = np.array(projected, dtype=np.float32).reshape(-1, 2)
    
    for i in range(len(corners)):
        # Convert to native Python integers
        real_point = (int(float(corners[i][0])), int(float(corners[i][1])))
        proj_point = (int(float(projected[i][0])), int(float(projected[i][1])))
        
        # Draw detected corners (green)
        cv2.circle(img_vis, real_point, 6, config.COLOR_DETECTED_CORNERS, -1)
        # Draw projected corners (red)
        cv2.circle(img_vis, proj_point, 4, config.COLOR_PROJECTED_CORNERS, -1)
        # Draw connection line
        cv2.line(img_vis, real_point, proj_point, config.COLOR_CONNECTION_LINES, 2)
        # Label corners
        label_point = (real_point[0] + 10, real_point[1])
        cv2.putText(img_vis, str(i), label_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw 3D coordinate axes
    axis_length = config.MARKER_SIZE_METERS
    axis_3d = np.array([
        [0, 0, 0],           # Origin
        [axis_length, 0, 0], # X-axis (red)
        [0, axis_length, 0], # Y-axis (green)
        [0, 0, -axis_length] # Z-axis (blue)
    ], dtype=np.float32)
    
    axis_projected, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
    axis_projected = axis_projected.reshape(-1, 2)
    
    # Convert to proper integer tuples with robust handling
    try:
        origin = (int(float(axis_projected[0][0])), int(float(axis_projected[0][1])))
        x_end = (int(float(axis_projected[1][0])), int(float(axis_projected[1][1])))
        y_end = (int(float(axis_projected[2][0])), int(float(axis_projected[2][1])))
        z_end = (int(float(axis_projected[3][0])), int(float(axis_projected[3][1])))
    except (IndexError, ValueError, TypeError) as e:
        print(f"[DEBUG] Axis coordinate conversion error: {e}")
        return  # Skip axis drawing if conversion fails
    
    # Draw axes using config colors
    cv2.arrowedLine(img_vis, origin, x_end, config.COLOR_X_AXIS, 3)  # X-axis
    cv2.arrowedLine(img_vis, origin, y_end, config.COLOR_Y_AXIS, 3)  # Y-axis
    cv2.arrowedLine(img_vis, origin, z_end, config.COLOR_Z_AXIS, 3)  # Z-axis
    
    # Add axis labels
    cv2.putText(img_vis, 'X', x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_X_AXIS, 2)
    cv2.putText(img_vis, 'Y', y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_Y_AXIS, 2)
    cv2.putText(img_vis, 'Z', z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_Z_AXIS, 2)
    
    # Display pose information
    distance = np.linalg.norm(tvec)
    cv2.putText(img_vis, f'Distance: {distance:.2f}m', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Pose Estimation Visualization", img_vis)
    cv2.waitKey(config.VISUALIZATION_DELAY)

def save_results_to_csv(filename, per_image_results, global_rmse, global_mean):
    """
    Save pose estimation results to CSV file
    
    Args:
        filename (str): Output CSV filename
        per_image_results (list): List of result dictionaries
        global_rmse (float): Global RMSE value
        global_mean (float): Global mean error
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header information
            writer.writerow(["Pose Estimation Results"])
            writer.writerow(["Timestamp", timestamp])
            writer.writerow(["Global RMSE (pixels)", f"{global_rmse:.6f}"])
            writer.writerow(["Global Mean Error (pixels)", f"{global_mean:.6f}"])
            writer.writerow(["Total Markers", len(per_image_results)])
            writer.writerow([])  # Empty row
            
            # Write detailed results header
            writer.writerow([
                "Filename", "Marker_ID", "RMSE", "Mean_Error", 
                "X_Position", "Y_Position", "Z_Position",
                "Rotation_X", "Rotation_Y", "Rotation_Z"
            ])
            
            # Write per-image results
            for result in per_image_results:
                rvec_deg = np.degrees(result['rvec'].flatten())
                tvec_flat = result['tvec'].flatten()
                
                writer.writerow([
                    result['filename'],
                    result['marker_id'],
                    f"{result['rmse']:.6f}",
                    f"{result['mean_error']:.6f}",
                    f"{tvec_flat[0]:.6f}",
                    f"{tvec_flat[1]:.6f}",
                    f"{tvec_flat[2]:.6f}",
                    f"{rvec_deg[0]:.3f}",
                    f"{rvec_deg[1]:.3f}",
                    f"{rvec_deg[2]:.3f}"
                ])
        
        print(f"[INFO] Results saved to {filename}")
        
    except Exception as e:
        print(f"[ERROR] Failed to save results to CSV: {e}")

if __name__ == "__main__":
    # Create configuration - you can choose between different presets
    # config = PoseEstimationPresets.get_high_precision_config()
    # config = PoseEstimationPresets.get_fast_processing_config()
    # config = PoseEstimationPresets.get_debug_config()
    # config = PoseEstimationPresets.get_large_marker_config()
    
    # Use default configuration
    config = PoseEstimationConfig()
    
    print("="*60)
    print("ARUCO MARKER POSE ESTIMATION")
    print("="*60)
    print(f"Calibration file: {config.get_absolute_calibration_path()}")
    print(f"Images directory: {config.get_absolute_images_path()}")
    print(f"Marker size: {config.MARKER_SIZE_METERS}m")
    print(f"ArUco dictionary: {config.ARUCO_DICTIONARY}")
    print()
    
    # Run pose estimation
    results, errors = estimate_pose_from_markers(config)
    
    # Analyze results
    analyze_pose_estimation_results(results, errors, config)
    
    if config.SHOW_VISUALIZATION:
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
    cv2.destroyAllWindows()