import torch
import cv2
from scipy.ndimage import label
import numpy as np

import matplotlib.pyplot as plt

def read_video_to_tensor(path: str, frame_step: int = 10, max_frame: int = 10000000) -> torch.Tensor:
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
        if frame_index >= max_frame:
            break

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

def heuristic_marker_highlight_for_first_frame(frame, threshold=0.5, ratio_threshold=3, marker_size_threshold=40, number_of_markers=3, picture=False):
    black_pixel_indices = torch.nonzero((frame < threshold).all(dim=0), as_tuple=False)

    mask = torch.zeros(frame.shape[1:], dtype=torch.uint8)
    mask[black_pixel_indices[:, 0], black_pixel_indices[:, 1]] = 1
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

    print(f"{black_pixel_indices.shape[0]} Black Pixels, ", f"{num_features} Segments.")

    marker = {}
    for key in segments.keys():
        marker[key] = torch.mean(torch.tensor(segments[key]).to(dtype=torch.float32), dim=0)

    if picture:
        highlighted_frame = frame.clone()
        for i in range(number_of_markers):
            highlighted_frame[:, segments[f'Marker {i+1}'][:, 0], segments[f'Marker {i+1}'][:, 1]] = torch.tensor([1.0, 0.0, 0.0]).view(3, 1)
            

        return marker, highlighted_frame
    return marker

def marker_highlight(frame, previous_marker, threshold=0.5, pixel_per_marker=50, picture=False):
    marker = {}
    
    black_pixel_indices = torch.nonzero((frame < threshold).all(dim=0), as_tuple=False)
    mask = torch.zeros(frame.shape[1:], dtype=torch.uint8)
    mask[black_pixel_indices[:, 0], black_pixel_indices[:, 1]] = 1
    
    for key in previous_marker.keys():
        coord = previous_marker[key]
        diff = black_pixel_indices.to(torch.float32) - coord
        distances = torch.sqrt((diff ** 2).sum(dim=1))
        sorted_indices = torch.argsort(distances)
        closest_indices = black_pixel_indices[sorted_indices[:pixel_per_marker]]
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


def rotation_from_projections_assuming_perfect_2d_camera(proj1: torch.Tensor, proj2: torch.Tensor) -> torch.Tensor:
    """
    Given:
      proj1: 2×3 tensor of orthographic projections of 3 non-collinear 3D vectors
             onto plane 1 (each column is one vector's (x,y) in plane 1 coords)
      proj2: 2×3 tensor of the same 3 vectors projected onto plane 2
    Returns:
      R: 3×3 rotation matrix mapping plane-1’s normal to plane-2’s normal
    """
    H = proj2 @ torch.pinverse(proj1)    

    U, S, Vt = torch.svd(H)
    H_ortho = U @ Vt                        
    
    if torch.det(H_ortho) < 0:
        Vt[-1, :] *= -1
        H_ortho = U @ Vt

    phi = torch.atan2(H_ortho[1,0], H_ortho[0,0])

    M = H_ortho - torch.eye(2)
    _, _, Vt2 = torch.svd(M)
    u2 = Vt2[:, -1]
    u2 = u2 / u2.norm()

    k = torch.tensor([u2[0], u2[1], 0.0], dtype=proj1.dtype, device=proj1.device)
    k = k / k.norm()

    K = torch.tensor([[    0,   -k[2],   k[1]],
                      [ k[2],      0,  -k[0]],
                      [-k[1],   k[0],      0]],
                     dtype=k.dtype, device=k.device)

    I3 = torch.eye(3, dtype=k.dtype, device=k.device)
    R = (I3 * torch.cos(phi)
         + (1 - torch.cos(phi)) * k.unsqueeze(1) @ k.unsqueeze(0)
         + torch.sin(phi) * K)

    return R