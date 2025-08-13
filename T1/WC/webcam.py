import numpy as np
import cv2 as cv
import os
import time
from datetime import datetime

def generate_filename(image_count, image_format="jpg"):
    """
    Generate filename with timestamp and counter
    
    Args:
        image_count (int): Sequential number of the image
        image_format (str): Image file format (jpg, png, etc.)
    
    Returns:
        str: Generated filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds precision
    filename = f"image_{image_count:06d}_{timestamp}.{image_format}"
    return filename

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

def capture_images_per_second(images_per_second=10, duration_seconds=60, output_dir="/home/woong/SIGMAxPortal301-Team2/T1/WC/pose estimation/marker_images", show_preview=True):
    """
    Capture a specified number of images per second from webcam
    
    Args:
        images_per_second (int): Number of images to capture per second (default: 10)
        duration_seconds (int): Total duration to capture images in seconds (default: 60)
        output_dir (str): Directory to save captured images (default: "captured_images")
        show_preview (bool): Whether to show live preview (default: True)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    # Calculate the interval between captures
    capture_interval = 1.0 / images_per_second  # seconds between each capture
    
    print(f"Starting capture: {images_per_second} images per second for {duration_seconds} seconds")
    print(f"Total images to capture: {images_per_second * duration_seconds}")
    print("Press 'q' to stop early")
    
    # 10 second delay before starting capture
    print("Starting capture in 10 seconds...")
    time.sleep(10)
    print("Capture started!")
    
    start_time = time.time()
    last_capture_time = start_time
    image_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam")
            break
        
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Check if it's time to capture an image
        if current_time - last_capture_time >= capture_interval:
            # Generate filename
            filename = generate_filename(image_count)
            filepath = os.path.join(output_dir, filename)
            
            # Save the image
            cv.imwrite(filepath, frame)
            image_count += 1
            last_capture_time = current_time
            
            print(f"Captured image {image_count}: {filename}")
        
        # Display the frame if enabled
        if show_preview:
            cv.imshow('Webcam Capture', frame)
        
        # Check for exit conditions
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user")
            break
        
        if elapsed_time >= duration_seconds:
            print(f"Capture duration ({duration_seconds}s) completed")
            break
    
    cap.release()
    cv.destroyAllWindows()
    
    print(f"\nCapture completed!")
    print(f"Total images captured: {image_count}")
    print(f"Images saved in: {output_dir}")
    print(f"Actual capture rate: {image_count / elapsed_time:.2f} images/second")

def capture_images_continuously(images_per_second=1, output_dir="captured_images", show_preview=True, progress_interval=10):
    """
    Capture images continuously until stopped manually
    
    Args:
        images_per_second (int): Number of images to capture per second (default: 10)
        output_dir (str): Directory to save captured images (default: "captured_images")
        show_preview (bool): Whether to show live preview (default: True)
        progress_interval (int): Print progress every N images (default: 10)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    # Calculate the interval between captures
    capture_interval = 1.0 / images_per_second  # seconds between each capture
    
    print(f"Starting continuous capture: {images_per_second} images per second")
    print("Press 'q' to stop")
    
    # 5 second delay before starting capture
    print("Starting capture in 5 seconds...")
    time.sleep(5)
    print("Capture started!")
    
    start_time = time.time()
    last_capture_time = start_time
    image_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam")
            break
        
        current_time = time.time()
        
        # Check if it's time to capture an image
        if current_time - last_capture_time >= capture_interval:
            # Generate filename
            filename = generate_filename(image_count)
            filepath = os.path.join(output_dir, filename)
            
            # Save the image
            cv.imwrite(filepath, frame)
            image_count += 1
            last_capture_time = current_time
            
            # Print progress based on interval
            if image_count % progress_interval == 0:
                elapsed_time = current_time - start_time
                print(f"Captured {image_count} images (Rate: {image_count / elapsed_time:.2f} imgs/sec)")
        
        # Display the frame if enabled
        if show_preview:
            cv.imshow('Webcam Continuous Capture', frame)
        
        # Check for exit condition
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user")
            break
    
    cap.release()
    cv.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    print(f"\nCapture completed!")
    print(f"Total images captured: {image_count}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Images saved in: {output_dir}")
    print(f"Average capture rate: {image_count / elapsed_time:.2f} images/second")

if __name__ == "__main__":
    # Choose one of the following options:
    
    # Option 1: Capture 10 images per second for 60 seconds
    # capture_images_per_second(images_per_second=10, duration_seconds=60, output_dir="captured_images")
    
    # Option 2: Capture images continuously until stopped manually (save to marker_images)
    capture_images_continuously(output_dir="marker_images")
    
    # Option 3: Custom settings example
    # capture_images_per_second(images_per_second=5, duration_seconds=30, output_dir="custom_output", show_preview=False)
    
    # Option 4: Original video recording functionality
    # writeVideoToFile()
    
    # Option 5: Just show webcam feed
    # capture_webcam_image()
