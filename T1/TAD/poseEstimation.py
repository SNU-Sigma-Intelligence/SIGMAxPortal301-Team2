import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def reprError(cameraMatrixPath, area, imagesPath):

    # Known intrinsics from calibration
    paramPath = os.path.join(cameraMatrixPath, "camera_calibration_params.npz")
    # print("Exists:", os.path.exists(paramPath))
    data = np.load(paramPath)
    K = data['camMatrix']                    # 3x3
    distCoeffs = data['distCoeff']          # (k1,k2,p1,p2,k3) or empty if undistorted
    patternSize = (9, 6)

  
    square_size = area  # meters
    objp = np.zeros((patternSize[0]*patternSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2) * square_size

    #Termination criteria for cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    image_paths = glob.glob(os.path.join(imagesPath, "frame*.jpg"))

    perImageRmse = []
    all_errors = []

    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] Could not read {p}, skipping.")
            continue    
    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        found, corners = cv2.findChessboardCornersSB(
        gray, patternSize,
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)

        if not found:
    
            # Fallback: classic method with preprocessing
            flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
            cv2.CALIB_CB_NORMALIZE_IMAGE |
            cv2.CALIB_CB_FILTER_QUADS)
            ok, corners = cv2.findChessboardCorners(gray, patternSize, flags)
            
            if(not ok):
                print("[INFO] Pattern not found")

        # Subpixel refine
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    
        ok, rvec, tvec = cv2.solvePnP(objp, corners, K, distCoeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE if patternSize[1] == patternSize[0] else cv2.SOLVEPNP_IPPE)
        if not ok:
            print(f"[WARN] solvePnP failed for {os.path.basename(p)}, skipping.")
            continue
        
        # print("Rotation matrix: ", rvec)
        # print("translation vector: ", tvec)
        
        # Reproject and compute errors
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, distCoeffs)  # Nx1x2
        proj = proj.reshape(-1, 2)
        obs  = corners.reshape(-1, 2)

        # per-point Euclidean pixel error
        errs = np.linalg.norm(proj - obs, axis=1)  # shape (N,)
        rmse = np.sqrt(np.mean(errs**2))

        perImageRmse.append((os.path.basename(p), rmse, errs.mean()))
        all_errors.append(errs)

        #visualizing reprojection 
        visualize_pose_check(img, corners, proj)
    
    return perImageRmse, all_errors

        
        

def computeReprError(perImageRmse, all_errors):
    if perImageRmse:
            all_errors = np.concatenate(all_errors)  # all points across all images
            global_rmse   = np.sqrt(np.mean(all_errors**2))
            global_mean   = all_errors.mean()
            # global_median = np.median(all_errors)

            print("\nPer-image error (pixels):  filename | RMSE | mean")
            for fname, rmse, mean_e in perImageRmse:
                print(f"{fname:>20s}  | {rmse:6.3f} | {mean_e:6.3f}")

            print("\nOverall across all images/points:")
            print(f"  Global RMSE : {global_rmse:.3f} px")
            print(f"  Global mean : {global_mean:.3f} px")
            # print(f"  Global median: {global_median:.3f} px")
            
            #save the global RMSE, mean error to an excel file
            fileName = r"C:\Users\tad1i\projects\SIGMAxPortal301-Team2\T1\TAD\chessboard pictures\reprojection Errors.txt" 
            save_repErrorsToText(fileName, global_rmse, global_mean)


          
    else:
        print("No valid poses computed.")


def visualize_pose_check(img, corners, proj):
    """
    img     : BGR image
    corners : detected corner points (Nx1x2 or Nx2)
    proj    : reprojected corner points (Nx1x2 or Nx2)
    """

    # Ensure shapes are (N, 2)
    real_pts = corners.reshape(-1, 2)
    proj_pts = proj.reshape(-1, 2)

    # Draw detected corners in green
    for pt in real_pts:
        cv2.circle(img, tuple(np.int32(pt)), 5, (0, 255, 0), -1)

    # Draw reprojected points in red
    for pt in proj_pts:
        cv2.circle(img, tuple(np.int32(pt)), 5, (0, 0, 255), -1)

    # Show overlay
    cv2.imshow("Corners vs. Projections", img)

    #Wait until 'q' is pressed
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    # OPTIONAL: Scatter plot (for reports / debugging)
    # plt.figure(figsize=(6, 6))
    # plt.scatter(real_pts[:, 0], real_pts[:, 1], c='g', marker='o', label='Detected Corners')
    # plt.scatter(proj_pts[:, 0], proj_pts[:, 1], c='r', marker='x', label='Reprojected Points')
    # plt.gca().invert_yaxis()
    # plt.xlabel("Pixel X")
    # plt.ylabel("Pixel Y")
    # plt.legend()
    # plt.title("Detected vs. Reprojected Corners")
    # plt.show()


def save_repErrorsToText(filename, gRMSE, gMeanError):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filename, "a") as f:
        f.write(f"Time - {timestamp}\n")
        f.write(f"  Global RMSE       : {gRMSE:.6f} px\n")
        f.write(f"  Global Mean Error : {gMeanError:.6f} px\n")
        f.write("-" * 40 + "\n")

    print(f"[INFO] Results saved to {filename}")