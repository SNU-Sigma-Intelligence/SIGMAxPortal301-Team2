import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment

def detect_blobs(image, min_area):
    params = cv.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = 5000
    params.filterByCircularity = True
    params.minCircularity = 0.7
    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return points, keypoints


image = cv.imread("marker.png")
gray_init = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
init_points, _ = detect_blobs(gray_init, min_area=15)
init_sorted = init_points[np.argsort(init_points[:, 1] + init_points[:, 0])]
prev_points = init_sorted.copy()

object_points = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0]
], dtype=np.float32)

h, w = gray_init.shape
f = w
camera_matrix = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5)

aligned_points = prev_points
success, rvec, tvec = cv.solvePnP(object_points, aligned_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_SQPNP)


R0, _ = cv.Rodrigues(rvec)
t0 = tvec.copy()

cap = cv.VideoCapture("marker.mp4")
cap.set(cv.CAP_PROP_POS_FRAMES, 0)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    points, _ = detect_blobs(gray, min_area=150)

    cv.putText(frame, f"Frame: {frame_count}", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1, (255, 0, 0), 2, cv.LINE_AA)

    if len(points) == 3:
        print(f"[FRAME {frame_count}] ▶ Detected Points:")
        for i, pt in enumerate(points):
            print(f"    pt{i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")

        aligned_points = match_points_hungarian(prev_points, points)
        success, rvec, tvec = cv.solvePnP(object_points, aligned_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_SQPNP)

        if success:
            R, _ = cv.Rodrigues(rvec)
            R_rel = R @ R0.T
            t_rel = tvec - R_rel @ t0
            print(f"[RELATIVE] r_rel:\n{R_rel}\nt_rel:\n{t_rel.ravel()}")

            for i, pt in enumerate(aligned_points):
                pt_int = tuple(np.int32(pt))
                cv.putText(frame, str(i + 1), (pt_int[0] + 10, pt_int[1] - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            prev_points = aligned_points.copy()
    else:
        print(f"[FRAME {frame_count}] ❌ Detected {len(points)}")

    cv.imshow("Pose Estimation", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
