import torch
import cv2
from scipy.ndimage import label
import numpy as np

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

def heuristic_marker_highlight(frame, threshold=0.5, number_of_marker=3, picture=False):
    black_pixel_indices = torch.nonzero((frame < threshold).all(dim=0), as_tuple=False)

    mask = torch.zeros(frame.shape[1:], dtype=torch.uint8)
    mask[black_pixel_indices[:, 0], black_pixel_indices[:, 1]] = 1
    labeled, num_features = label(mask.numpy())
    segments = {}
    for seg_id in range(1, num_features + 1):
        indices = np.argwhere(labeled == seg_id)
        segments[f"segment {seg_id}"] = indices

    bboxes = {}
    height, width = frame.shape[1], frame.shape[2]
    for seg_id, indices in segments.items():
        ymin, xmin = np.min(indices, axis=0)
        ymax, xmax = np.max(indices, axis=0)
        ext_ymin = max(0, ymin - 30)
        ext_xmin = max(0, xmin - 30)
        ext_ymax = min(height - 1, ymax + 30)
        ext_xmax = min(width - 1, xmax + 30)
        bboxes[seg_id] = {
            'bbox': (ymin, xmin, ymax, xmax),
            'extended_bbox': (ext_ymin, ext_xmin, ext_ymax, ext_xmax)
        }
    
    for seg_id, props in bboxes.items():
        ext_ymin, ext_xmin, ext_ymax, ext_xmax = props['extended_bbox']
        region = frame[:, ext_ymin:ext_ymax+1, ext_xmin:ext_xmax+1]
        avg_whiteness = region.mean().item()
        bboxes[seg_id]['avg_whiteness'] = avg_whiteness
    sorted_segments = sorted(bboxes.items(), key=lambda x: x[1]['avg_whiteness'], reverse=True)

    if len(sorted_segments) >= number_of_marker:
        name_order = [f'Marker {i}' for i in range(1, number_of_marker + 1)]
        segments = {name: segments[seg_id] for name, (seg_id, _) in zip(name_order, sorted_segments[:3])}
    else:
        raise ValueError("Not enough segments found to assign O, A, B.")
    
    print(f"{black_pixel_indices.shape[0]} Black Pixels, ", f"{num_features} Segments.")

    if picture:
        highlighted_frame = frame.clone()
        for i in range(number_of_marker):
            highlighted_frame[:, segments[f'Marker {i+1}'][:, 0], segments[f'Marker {i+1}'][:, 1]] = torch.tensor([1.0, 0.0, 0.0]).view(3, 1)

        return segments, highlighted_frame
    return segments
