from typing import Any, List, Dict, Optional, Tuple
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from pearl.agent import RLAgent
from pearl.env import RLEnvironment
from pearl.method import ExplainabilityMethod

from pearl.lab.visual import VisualizationMethod
from pearl.lab.annotations import Param

class StabilityVisualizationParams:
    action: Param(int) = 0

class NoiseGenerator:
    """Handles different types of noise generation for tensors."""

    @staticmethod
    def add_gaussian_noise(tensor: torch.Tensor, std: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise to tensor and clamp to [0, 1]."""
        noise = torch.normal(mean=0.0, std=std, size=tensor.shape, device=tensor.device)
        return (tensor + noise).clamp(0.0, 1.0)

    @staticmethod
    def add_salt_pepper_noise(tensor: torch.Tensor, amount: float = 0.01) -> torch.Tensor:
        """Add salt and pepper noise to tensor."""
        noisy = tensor.clone()
        has_batch = tensor.dim() == 4

        if has_batch:
            batch_size = tensor.shape[0]
            elements_per_item = tensor.numel() // batch_size

            for b in range(batch_size):
                NoiseGenerator._apply_salt_pepper_to_item(
                    noisy[b], elements_per_item, amount
                )
        else:
            NoiseGenerator._apply_salt_pepper_to_item(
                noisy, tensor.numel(), amount
            )

        return noisy

    @staticmethod
    def _apply_salt_pepper_to_item(tensor: torch.Tensor, total_elements: int, amount: float):
        """Apply salt and pepper noise to a single tensor item."""
        num_pixels = int(amount * total_elements)
        if num_pixels == 0:
            return

        indices = torch.randperm(total_elements, device=tensor.device)[:num_pixels]
        flat_tensor = tensor.view(-1)

        # Randomly set to 0 or 1
        values = torch.randint(0, 2, (num_pixels,), dtype=tensor.dtype, device=tensor.device)
        flat_tensor[indices] = values


class Visualizer:
    """Handles visualization of observations and noisy observations."""

    def __init__(self, save_dir: str = "./temp"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self._image_counter = 0

    def save_comparison(self,
                       original: np.ndarray,
                       noisy: np.ndarray,
                       title: str = "Original vs Noisy") -> None:
        """Save side-by-side comparison of original and noisy observations."""
        if original.ndim != 3 or noisy.ndim != 3:
            raise ValueError("Expected 3D arrays (C, H, W)")

        num_frames = original.shape[0]
        fig, axes = plt.subplots(2, num_frames, figsize=(2 * num_frames, 4))

        # Handle single frame case
        if num_frames == 1:
            axes = axes.reshape(2, 1)

        for i in range(num_frames):
            # Original frames
            axes[0, i].imshow(original[i], cmap='gray', vmin=0, vmax=255)
            axes[0, i].set_title(f"Original {i+1}")
            axes[0, i].axis('off')

            # Noisy frames
            axes[1, i].imshow(noisy[i], cmap='gray', vmin=0, vmax=255)
            axes[1, i].set_title(f"Noisy {i+1}")
            axes[1, i].axis('off')

        plt.suptitle(title)
        plt.tight_layout()

        save_path = self.save_dir / f"obs_{self._image_counter}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self._image_counter += 1


class StabilityExplainability(ExplainabilityMethod):
    """
    Explainability method that measures agent stability under input noise.

    Evaluates how consistent agent actions remain when noise is added to observations.
    Higher consistency scores indicate more robust/stable agents.
    """

    SUPPORTED_NOISE_TYPES = {'gaussian', 'salt_pepper'}

    def __init__(self,
                 device: torch.device,
                 noise_type: Optional[str] = 'gaussian',
                 noise_strength: Optional[float] = 0.01,
                 num_samples: Optional[int] = 20,
                 reward_weight: Optional[bool] = True,
                 visualize: Optional[bool] = True,
                 save_dir: Optional[str] = "./temp"):
        """
        Initialize stability explainability method.

        Args:
            device: PyTorch device for computations
            noise_type: Type of noise ('gaussian' or 'salt_pepper')
            noise_strength: Strength of noise (std for gaussian, amount for salt_pepper)
            num_samples: Number of noisy samples to test per observation
            visualize: Whether to save visualization of first observation
            save_dir: Directory to save visualizations
        """
        super().__init__()

        if noise_type not in self.SUPPORTED_NOISE_TYPES:
            raise ValueError(f"Unsupported noise type: {noise_type}. "
                           f"Supported: {self.SUPPORTED_NOISE_TYPES}")

        self.device = device
        self.noise_type = noise_type
        self.noise_strength = noise_strength
        self.num_samples = num_samples

        self.agent: Optional[List[RLAgent]] = None
        self.env: Optional[RLEnvironment] = None

        self.visualizer = Visualizer(save_dir) if visualize else None
        self._first_observation = True
        self.prev_stability_score = 0
        self.current_stability_score = 0
        self.last_reward: Optional[Dict[str, np.ndarray]] = None
        self.reward_weight = reward_weight

    def set(self, env: RLEnvironment) -> None:
        """Set the environment."""
        super().set(env)
        self.env = env

    def prepare(self, agent: RLAgent) -> None:
        """Prepare with list of agents to evaluate."""
        self.agent = agent

    def onStep(self, action: Any) -> None:
        """Called before step - no action needed."""
        pass

    def onStepAfter(self, action: Any, reward: Dict[str, np.ndarray], done: bool, info: dict):
        # nothing for shap
        self.last_reward = reward

    def _apply_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the specified type of noise to the tensor."""
        if self.noise_type == 'gaussian':
            return NoiseGenerator.add_gaussian_noise(tensor, self.noise_strength)
        elif self.noise_type == 'salt_pepper':
            return NoiseGenerator.add_salt_pepper_noise(tensor, self.noise_strength)
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")

    def _tensor_to_numpy_uint8(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array in uint8 format for visualization."""
        return (tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

    def _measure_agent_stability(self, agent: RLAgent, obs_tensor: torch.Tensor) -> int:
        """Measure how many times an agent gives consistent actions under noise."""
        base_action = np.argmax(agent.predict(obs_tensor))
        consistent_count = 0

        for _ in range(self.num_samples):
            noisy_obs = self._apply_noise(obs_tensor)
            noisy_action = np.argmax(agent.predict(noisy_obs))

            if noisy_action == base_action:
                consistent_count += 1

        return consistent_count

    def explain(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute stability scores for all agents on given observation.

        Args:
            obs: Observation array

        Returns:
            Array of consistency counts for each agent
        """
        if self.agent is None:
            raise RuntimeError("Must call prepare() with agents before explain()")

        obs_tensor = torch.as_tensor(obs, dtype=torch.float, device=self.device)

        # Visualize first observation if enabled
        if self.visualizer and self._first_observation:
            with torch.no_grad():
                noisy_tensor = self._apply_noise(obs_tensor)

                obs_np = self._tensor_to_numpy_uint8(obs_tensor)
                noisy_np = self._tensor_to_numpy_uint8(noisy_tensor)

                self.visualizer.save_comparison(obs_np, noisy_np, "Original vs Noisy")
                self._first_observation = False

        # Measure stability for each agent
        score = self._measure_agent_stability(self.agent, obs_tensor)
        return score

    def value(self, obs) -> float:
      self.current_stability_score = self.explain(obs)

      result = 0.0
      if self.reward_weight and self.last_reward is not None:
        result = self.prev_stability_score * float(self.last_reward['reward'])
      else:
        result = self.current_stability_score

      # Always update the previous score for next iteration
      self.prev_stability_score = self.current_stability_score

      return result

    def supports(self, m: VisualizationMethod) -> bool:
        return False

    def getVisualizationParamsType(self, m: VisualizationMethod) -> type | None:
        if not isinstance(m, VisualizationMethod):
            m = VisualizationMethod(m)

        if m == VisualizationMethod.HEAT_MAP:
            return StabilityVisualizationParams
        return None

    def getVisualization(self, m: VisualizationMethod, params: Any = None) -> np.ndarray | dict | None:
        pass
