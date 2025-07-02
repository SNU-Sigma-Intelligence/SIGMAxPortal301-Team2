import torch
import cv2
from scipy.ndimage import label
import numpy as np
from typing import Tuple

import matplotlib.pyplot as plt

def heuristic_marker_highlight_for_first_frame(frame, marker="red", 
                                               black_threshold=0.5, red_threshold=0.8, gb_threshold=0.2,
                                               ratio_threshold=3, marker_size_threshold=40, number_of_markers=3, picture=False):
    if marker == "black":
        pixel_indices = torch.nonzero((frame < black_threshold).all(dim=0), as_tuple=False)
    elif marker == "red":
        is_red = (frame[0] > red_threshold) & (frame[1] < gb_threshold) & (frame[2] < gb_threshold)
        pixel_indices = torch.nonzero(is_red, as_tuple=False)
    else:
        raise ValueError(f"Unsupported marker color: {marker}")

    mask = torch.zeros(frame.shape[1:], dtype=torch.uint8)
    mask[pixel_indices[:, 0], pixel_indices[:, 1]] = 1
    labeled, num_features = label(mask.numpy())
    segments = {}
    for seg_id in range(1, num_features + 1):
        indices = np.argwhere(labeled == seg_id)
        segments[f"segment {seg_id}"] = indices

    eligible_segments = {}
    for name, indices in segments.items():
        x_min, x_max = indices[:, 1].min(), indices[:, 1].max()
        y_min, y_max = indices[:, 0].min(), indices[:, 0].max()
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        if (width < marker_size_threshold and height < marker_size_threshold 
            and width < ratio_threshold * height and height < ratio_threshold * width):
            eligible_segments[name] = indices
    sorted_segments = sorted(eligible_segments.items(), key=lambda x: len(x[1]), reverse=True)

    renamed_segments = {}
    if len(sorted_segments) > number_of_markers:
        for i in range(number_of_markers):
            renamed_segments[f"Marker {i+1}"] = sorted_segments[i][1]

    segments = renamed_segments

    print(f"{pixel_indices.shape[0]} {marker} Pixels, ", f"{num_features} Segments.")

    marker = {}
    for key in segments.keys():
        marker[key] = torch.mean(torch.tensor(segments[key]).to(dtype=torch.float32), dim=0)

    if picture:
        highlighted_frame = frame.clone()
        for i in range(number_of_markers):
            highlighted_frame[:, segments[f'Marker {i+1}'][:, 0], segments[f'Marker {i+1}'][:, 1]] = torch.tensor([0.0, 1.0, 0.0]).view(3, 1)
            

        return marker, highlighted_frame
    return marker

def marker_highlight(frame, previous_marker, marker="red",
                      black_threshold=0.5, red_threshold=0.8, gb_threshold=0.2,
                      pixel_per_marker=50, picture=False):
    
    if marker == "black":
        pixel_indices = torch.nonzero((frame < black_threshold).all(dim=0), as_tuple=False)
    elif marker == "red":
        is_red = (frame[0] > red_threshold) & (frame[1] < gb_threshold) & (frame[2] < gb_threshold)
        pixel_indices = torch.nonzero(is_red, as_tuple=False)
    else:
        raise ValueError(f"Unsupported marker color: {marker}")
    
    marker = {}
    mask = torch.zeros(frame.shape[1:], dtype=torch.uint8)
    mask[pixel_indices[:, 0], pixel_indices[:, 1]] = 1
    
    for key in previous_marker.keys():
        coord = previous_marker[key]
        diff = pixel_indices.to(torch.float32) - coord
        distances = torch.sqrt((diff ** 2).sum(dim=1))
        sorted_indices = torch.argsort(distances)
        closest_indices = pixel_indices[sorted_indices[:pixel_per_marker]]
        marker[key] = closest_indices

    segments = marker
    marker = {}
    for key in segments.keys():
        marker[key] = torch.mean(torch.tensor(segments[key]).to(dtype=torch.float32), dim=0)

    if picture:
        highlighted_frame = frame.clone()
        for key in segments.keys():
            highlighted_frame[:, segments[key][:, 0], segments[key][:, 1]] = torch.tensor([1.0, 0.0, 0.0]).view(3, 1)
        return marker, highlighted_frame
    return marker

def show_picture_with_marker_vector(frame, marker, title="Title"):
    img = frame.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

    origin = marker.get("Marker 1")
    if origin is not None:
        origin_x, origin_y = origin[1].item(), origin[0].item()
        for key, value in marker.items():
            if key == "Marker 1":
                continue
            target_x, target_y = value[1].item(), value[0].item()
            dx = target_x - origin_x
            dy = target_y - origin_y
            plt.arrow(origin_x, origin_y, dx, dy, color='blue', linewidth=2, head_width=5, head_length=5)
            plt.text(origin_x + dx/2, origin_y + dy/2, f"1 to {key.split()[-1]}", color='blue', fontsize=12)

    plt.show()


def compute_projection_relation_assuming_perfect_2d_camera(
    X: torch.Tensor,
    Y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute in-plane 2×2 map M, angle θ, and intersection-line direction e.

    Y = M X
    M = Y X^T (X X^T)^{-1}
    θ = arccos(det(M))
    M e = e
    """
    XXT = X @ X.T
    M = Y @ X.T @ torch.inverse(XXT)
    detM = torch.det(M).clamp(-1.0, 1.0)
    theta = torch.acos(detM)
    eigvals, eigvecs = torch.linalg.eig(M)
    idx = torch.argmin((eigvals - 1.0).abs()).item()
    e = eigvecs[:, idx].real
    e = e / e.norm()
    return M, theta, e