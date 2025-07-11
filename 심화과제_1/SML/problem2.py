import torch
from scipy.ndimage import label
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize

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
            renamed_segments[f"Marker {i+1}"] = sorted_segments[i][1] #sorted_segments[i][1] is the indices of the segment

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


def compute_projection_relation_assuming_perfect_2d_camera(X: torch.Tensor, Y: torch.Tensor):
    """
    Fit y_i ≈ R(θ)·diag(a,b)·x_i in least-squares sense with θ free and a,b≥0.

    Parameters
    ----------
    X : torch.Tensor, shape (N,2)
    Y : torch.Tensor, shape (N,2)

    Returns
    -------
    theta : torch.Tensor
        Optimal rotation angle in radians.
    a : torch.Tensor
        Scale along the first axis (≥0).
    b : torch.Tensor
        Scale along the second axis (≥0).
    R : torch.Tensor, shape (2,2)
        Rotation matrix R(θ).
    D : torch.Tensor, shape (2,2)
        Diagonal scaling matrix diag(a,b).
    """
    X_np = X.cpu().double().numpy()
    Y_np = Y.cpu().double().numpy()
    denom_a = np.sum(X_np[:,0]**2)
    denom_b = np.sum(X_np[:,1]**2)

    def cost(theta):
        c, s = np.cos(theta), np.sin(theta)
        Rinv = np.array([[ c, s],[-s, c]])
        U = Y_np.dot(Rinv.T)
        a_ = np.sum(U[:,0] * X_np[:,0]) / denom_a
        b_ = np.sum(U[:,1] * X_np[:,1]) / denom_b
        a_ = max(0.0, a_)
        b_ = max(0.0, b_)
        R = np.array([[c, -s],[s, c]])
        Y_pred = (X_np * np.array([a_, b_])).dot(R.T)
        return np.sum((Y_np - Y_pred)**2)

    res = minimize_scalar(cost, bounds=(-np.pi, np.pi), method='bounded')
    theta_opt = res.x
    c, s = np.cos(theta_opt), np.sin(theta_opt)
    Rinv = np.array([[ c, s],[-s, c]])
    U_opt = Y_np.dot(Rinv.T)
    a_opt = np.sum(U_opt[:,0] * X_np[:,0]) / denom_a
    b_opt = np.sum(U_opt[:,1] * X_np[:,1]) / denom_b
    a_opt = max(0.0, a_opt)
    b_opt = max(0.0, b_opt)
    theta = torch.tensor(theta_opt, dtype=X.dtype, device=X.device)
    a = torch.tensor(a_opt, dtype=X.dtype, device=X.device)
    b = torch.tensor(b_opt, dtype=X.dtype, device=X.device)
    R = torch.tensor([[c, -s],[s, c]], dtype=X.dtype, device=X.device)
    D = torch.diag(torch.tensor([a_opt, b_opt], dtype=X.dtype, device=X.device))
    return theta * 180.0 / torch.pi, a, b, R, D

def fit_TR(X, Y):
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)

    def cost(vars):
        th_deg, ph_deg, r_deg = vars
        th = np.deg2rad(th_deg)
        ph = np.deg2rad(ph_deg)
        r  = np.deg2rad(r_deg)

        cth, sth = np.cos(th), np.sin(th)
        Rth = np.array([[ cth, -sth],
                        [ sth,  cth]])

        cph, sph = np.cos(ph), np.sin(ph)
        Rph = np.array([[ cph, -sph],
                        [ sph,  cph]])
        D   = np.diag([1.0, np.cos(r)])
        Tmat= Rph.dot(D).dot(Rph.T)

        Ypred = (X.dot(Rth.T)).dot(Tmat.T)
        return np.sum((Y - Ypred)**2)

    res = minimize(
      cost,
      x0=[0.0, 0.0, 0.0],
      bounds=[(0,360), (0,360), (0,180)],
      method='L-BFGS-B'
    )

    th_deg, ph_deg, r_deg = res.x.tolist()

    th, ph, r = np.deg2rad(th_deg), np.deg2rad(ph_deg), np.deg2rad(r_deg)
    R = np.array([[ np.cos(th), -np.sin(th)],
                  [ np.sin(th),  np.cos(th)]])
    Rph = np.array([[ np.cos(ph), -np.sin(ph)],
                    [ np.sin(ph),  np.cos(ph)]])
    T = Rph.dot(np.diag([1.0, np.cos(r)])).dot(Rph.T)

    return th_deg, ph_deg, r_deg, R, T

def fit_TR_analytic(X, Y):
    """
    Analytically fit Y ≈ T(φ,ρ)·R(θ)·X in least‑squares sense.

    Returns angle θ in degrees ∈ [-180,180], and φ, ρ in degrees ∈ [0,180],
    along with the 2×2 matrices R(θ) and T(φ,ρ).
    """
    X = np.asarray(X, float) 
    Y = np.asarray(Y, float) 
    
    XtX = X.T.dot(X)   
    W = np.linalg.inv(XtX).dot(X.T).dot(Y)  
    M = W.T                        

    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    sigma1, sigma2 = S  

    ratio = sigma2 / sigma1
    ratio = np.clip(ratio, -1.0, 1.0)

    phi = np.degrees(np.arctan2(U[1,0], U[0,0]))
    if phi > 180:
        phi -= 360
    if phi <= -180:
        phi += 360

    rho = np.degrees(np.arccos(ratio))

    R = U.dot(Vt)
    theta = np.degrees(np.arctan2(R[1,0], R[0,0]))

    ph = np.radians(phi)
    cph, sph = np.cos(ph), np.sin(ph)
    Rph = np.array([[ cph, -sph], [ sph,  cph]])
    D = np.diag([1.0, np.cos(np.radians(rho))])
    T = Rph.dot(D).dot(Rph.T)

    if theta > 180:
        theta -= 360
    if theta <= -180:
        theta += 360

    if phi < 0:
        phi += 180

    return float(theta), float(phi), float(rho), R, T
