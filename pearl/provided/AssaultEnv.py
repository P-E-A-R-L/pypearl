from typing import Any
import numpy as np
from pearl.mask import Mask


class AssaultEnvShapMask(Mask):
    """
    Simplified SHAP mask for the ALE/Assault environment.
    Computes action relevance scores based on overlap between non-zero observation pixels
    and the per-pixel attributions from SHAP.
    """

    def __init__(self):
        super().__init__(7)  # Assault has 7 actions
        self.fg_mask = None  # Foreground mask computed in update()

    def update(self, frame: np.ndarray):
        """
        frame: np.ndarray of shape (C, H, W) or (1, C, H, W)
        Converts the latest frame into a binary mask: 1 for non-zero pixels.
        """
        if frame.ndim == 4:
            frame = frame[0]  # drop batch dim
        last_frame = frame[-1] if frame.ndim == 3 else frame  # (H, W)
        self.fg_mask = (last_frame != 0).astype(np.float32)  # shape (H, W)
        # pixels under 5 in the y-axis are considered background (score)
        self.fg_mask[:5, :] = 0.0
        # lives, and the line under the ship are also background
        self.fg_mask[74:76, :] = 0.0
        self.fg_mask[74:, 0:30] = 0.0
        # take the pixels touching the foreground as foreground (morphological operation)
        self.fg_mask = np.where(self.fg_mask > 0, 1.0, 0.0)
        
        # output the mask as an image to assault_foreground_mask.png
        import matplotlib.pyplot as plt
        plt.imsave('assault_foreground_mask.png', self.fg_mask, cmap='gray', vmin=0, vmax=1)
        
        # punish focusing on background
        self.fg_mask[self.fg_mask == 0] = -0.5
        
    def compute(self, values: Any) -> np.ndarray:
        """
        values: np.ndarray of shape (1, T, H, W, A)
        Extracts values from the latest frame and multiplies with foreground mask.
        Returns score per action.
        """
        if values.ndim != 5:
            raise ValueError(f"Expected shape (1, T, H, W, A), got {values.shape}")

        _, T, H, W, A = values.shape
        last_vals = values[0, -1]  # shape (H, W, A)
        if self.fg_mask is None:
            return np.zeros(A, dtype=np.float32)

        scores = np.zeros(A, dtype=np.float32)
        for a in range(A):
            abs_map = np.abs(last_vals[..., a])
            abs_map = (abs_map - np.min(abs_map)) / (np.max(abs_map) - np.min(abs_map) + 1e-8)
            abs_map /= np.sum(abs_map) if np.sum(abs_map) > 0 else 1.0  # normalize to sum to 1
            scores[a] = np.sum(abs_map * self.fg_mask)
        return scores
