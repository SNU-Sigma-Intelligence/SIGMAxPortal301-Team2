import torch
import cv2

import matplotlib.pyplot as plt

def read_video_to_tensor(path: str, frame_step: int = 10) -> torch.Tensor:
    cap = cv2.VideoCapture(path)
    frames = []
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_index % frame_step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame_tensor)
        frame_index += 1

    cap.release()

    if not frames:
        raise ValueError("Failed to read any frames from video.")
        
    return torch.stack(frames)

def show_picture(tensor, title="Title"):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()