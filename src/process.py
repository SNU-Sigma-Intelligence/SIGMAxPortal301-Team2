import torch
import cv2
import numpy as np

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