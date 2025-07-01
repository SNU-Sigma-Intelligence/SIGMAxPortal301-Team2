import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def detect_blobs(image, setting):
    params = cv.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0  
    params.filterByArea = True
    params.minArea = setting
    params.maxArea = 5000
    params.filterByCircularity = True
    params.minCircularity = 0.7
    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return points, keypoints

image = cv.imread("gray image.png")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
firstpoints, firstkeypoints = detect_blobs(image,15)

for i, pt in enumerate(firstpoints):
    print(f"Point {i+1}: (x={pt[0]:.1f}, y={pt[1]:.1f})")
out_img = cv.drawKeypoints(gray, firstkeypoints, None, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(out_img, cmap='gray')
plt.title("Detected Blobs")
plt.axis("off")
plt.show() 

cap = cv.VideoCapture("realmarker.mp4") 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    points, keypoints = detect_blobs(gray, 45) # 자꾸 오른쪽 점 같이 잡는데 해결 필요함

    for i, pt in enumerate(points):
        print(f"  Point {i+1}: (x={pt[0]:.1f}, y={pt[1]:.1f})")

    vis_frame = cv.drawKeypoints(gray, keypoints, None, (0, 0, 255),
                                 cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv.imshow("Blob Detection", vis_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()  