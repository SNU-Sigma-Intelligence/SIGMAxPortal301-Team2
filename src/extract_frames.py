import cv2
import os

def extract_frames(video_path, output_dir, frames_per_second):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # Get the original frame rate of the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / original_fps

    print(f"Original FPS: {original_fps}, Duration: {duration_sec:.2f}s, Total frames: {total_frames}")

    # Calculate interval between frames to be extracted
    interval = int(original_fps / frames_per_second)
    if interval < 1:
        raise ValueError("Requested frames per second is higher than the video FPS.")

    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_idx % interval == 0:
            frame_filename = os.path.join(output_dir, f"frame{frame_idx:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"Saved {saved_count} frames to '{output_dir}'")


