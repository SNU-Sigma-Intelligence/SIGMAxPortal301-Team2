import numpy as np
import cv2 as cv
import os

def capture_webcam_image():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        exit()

    while True:
        ret, frame = cap.read()
        if ret:
            cv.imshow('Webcam', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def videoFromFile():
    root = os.getcwd()
    video_path = os.path.join(root, 'video.mp4')
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open video file")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv.imshow('Video', frame)
        delay = int(1000 / cap.get(cv.CAP_PROP_FPS)) if cap.get(cv.CAP_PROP_FPS) > 0 else 33

        if cv.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def writeVideoToFile():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        exit()

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    outpath = os.path.join(os.getcwd(), 'output.avi')
    out = cv.VideoWriter(outpath, fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv.imshow('Recording', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    writeVideoToFile()
