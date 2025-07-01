import torch
import cv2
import numpy as np

from utils import Camera

def perform_PnP(camera: Camera, points_3D: torch.Tensor, points_2D: torch.Tensor) -> torch.Tensor:
    """
    Performs Perspective-n-Point (PnP) algorithm to estimate the pose of a camera given 3D points and their corresponding 2D projections.

    Args:
        camera (Camera): Camera object containing intrinsic parameters.
        points_3D (torch.Tensor): Tensor of shape [N, 3] representing 3D points in the world coordinate system.
        points_2D (torch.Tensor): Tensor of shape [N, 2] representing 2D projections of the 3D points in the image plane.

    Returns:
        torch.Tensor: Rotation vector and translation vector as a tensor of shape [3, 4].
    """
    K = camera.calibration_matrix.cpu().numpy()
    dist_coeffs = camera.distortion_coeffs.cpu().numpy()

    points_3D_np = points_3D.cpu().numpy()
    points_2D_np = points_2D.cpu().numpy()

    success, rvec, tvec = cv2.solvePnP(points_3D_np, points_2D_np, K, dist_coeffs)
    
    if not success:
        raise RuntimeError("PnP failed to find a solution")

    R, _ = cv2.Rodrigues(rvec)
    pose = np.hstack((R, tvec.reshape(-1, 1)))
    
    return torch.from_numpy(pose).float()

def extract_sift_features(image: torch.Tensor) -> torch.Tensor:
    """
    Extracts SIFT features from an RGB image tensor.

    Args:
        image (torch.Tensor): Input image tensor of shape [3, H, W], values in [0, 1].

    Returns:
        torch.Tensor: Tensor of shape [N, 128], where N is the number of detected keypoints.
                      Returns an empty tensor with shape [0, 128] if no keypoints are found.
    """
    image_np = image.mul(255).byte().cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(image_gray, None)
    descriptors = np.zeros((0, 128), dtype=np.float32) if descriptors is None else descriptors
    return torch.from_numpy(descriptors)