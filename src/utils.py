import torch
import cv2
from scipy.ndimage import label
import numpy as np
from typing import Tuple

import matplotlib.pyplot as plt

import torch

class Camera:
    def __init__(self,
                 focal_length: torch.Tensor = None,
                 sensor_size: torch.Tensor = None,
                 principal_point: torch.Tensor = None,
                 calibration_matrix: torch.Tensor = None,
                 distortion_coeffs: torch.Tensor = None,
                 rotation_matrix: torch.Tensor = None,
                 translation_vector: torch.Tensor = None):
        """
        Initializes a Camera object with intrinsic and extrinsic parameters.
        All parameters are expected as torch.Tensors.

        Parameters:
        - focal_length: Tensor of shape (1,) or scalar tensor
        - sensor_size: Tensor of shape (2,) (width, height)
        - principal_point: Tensor of shape (2,)
        - calibration_matrix: Tensor of shape (3, 3)
        - distortion_coeffs: Tensor of shape (N,), e.g. [k1, k2, p1, p2, k3]
        - rotation_matrix: Tensor of shape (3, 3)
        - translation_vector: Tensor of shape (3,)
        """
        self.focal_length = focal_length
        self.sensor_size = sensor_size
        self.principal_point = principal_point
        self.calibration_matrix = calibration_matrix
        self.distortion_coeffs = distortion_coeffs
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector

    def set_intrinsics(self, focal_length: torch.Tensor, sensor_size: torch.Tensor, principal_point: torch.Tensor):
        self.focal_length = focal_length
        self.sensor_size = sensor_size
        self.principal_point = principal_point

    def set_calibration_matrix(self, K: torch.Tensor):
        if K.shape == (3, 3):
            self.calibration_matrix = K
        else:
            raise ValueError("Calibration matrix must be 3x3")

    def set_distortion(self, dist: torch.Tensor):
        self.distortion_coeffs = dist

    def set_extrinsics(self, R: torch.Tensor, T: torch.Tensor):
        if R.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
        if T.shape != (3,) and T.shape != (3, 1):
            raise ValueError("Translation vector must be of shape (3,) or (3,1)")
        self.rotation_matrix = R
        self.translation_vector = T.view(3)

    def get_projection_matrix(self) -> torch.Tensor:
        """
        Returns the 3x4 projection matrix P = K * [R | T]
        """
        if self.calibration_matrix is None or self.rotation_matrix is None or self.translation_vector is None:
            raise ValueError("Missing calibration or extrinsic parameters")
        RT = torch.cat((self.rotation_matrix, self.translation_vector.view(3, 1)), dim=1)
        return self.calibration_matrix @ RT

    def __repr__(self):
        return f"Camera(focal_length={self.focal_length}, sensor_size={self.sensor_size}, principal_point={self.principal_point})"

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