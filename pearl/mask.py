from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Mask(ABC):
    """
    Simple abstract base class for masking actions in reinforcement learning.
    """

    def __init__(self, action_space=None):
        """
        Initialize the mask.

        Args:
            action_space: The action space of the environment
        """
        self.action_space = action_space

    @abstractmethod
    def update(self, observation: Any):
        """
        Update the mask based on the current observation.

        Args:
            observation: The current observation from the environment
                         (N, H, W, C)
        """
        pass

    @abstractmethod
    def compute(self, values: Any) -> np.ndarray:
        """
        Returns the value of the mask for the given values.

        Args:
            values: The current values to multiply by the mask
                    Expects values to be in the same shape as observations as in update function
                    (N, H, W, C, Action)

        Returns:
            mask: tensor of shape [actions]
        """
        pass