from typing import Any, Dict
import numpy as np
import shap
import torch
from skimage.color import gray2rgb
from pearl.agent import RLAgent
from pearl.env import RLEnvironment
from pearl.mask import Mask
from pearl.method import ExplainabilityMethod

from pearl.lab.visual import VisualizationMethod
from pearl.lab.annotations import Param

# A little hack to fix the issue of shap not being able to handle Flatten layer
from shap.explainers._deep import deep_pytorch
deep_pytorch.op_handler['Flatten'] = deep_pytorch.passthrough

class ShapVisualizationParams:
    mode: Param(str, choices=["Last Action", "Selected Action"]) = "Last Action"
    action: Param(int) = 0
    threshold: Param(float, choices=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]) = 0.4

class ShapExplainability(ExplainabilityMethod):
    def __init__(self, device, mask: Mask):
        super().__init__()
        self.device = device
        self.explainer = None
        self.background = None
        self.mask = mask
        self.agent = None
        self.last_explain = None
        self.last_obs = None

    def set(self, env: RLEnvironment):
        super().set(env)
        self.background = torch.zeros(env.get_observations().shape).to(self.device)

    def prepare(self, agent: RLAgent):
        model = agent.get_q_net().to(self.device)
        self.explainer = shap.DeepExplainer(model, self.background)
        self.agent = agent

    def onStep(self, action: Any):
        self.last_action = action

    def onStepAfter(self, action: Any, reward: Dict[str, np.ndarray], done: bool, info: dict):
        # nothing for shap
        pass

    def explain(self, obs) -> np.ndarray | Any:
        if self.explainer is None:
            raise ValueError("Explainer not set. Please call prepare() first.")
        
        obs_tensor = torch.as_tensor(obs, dtype=torch.float, device=self.device)
        shap_values = self.explainer.shap_values(obs_tensor, check_additivity=False)
        
        # stored for visualization
        self.last_obs = obs
        self.last_explain = shap_values
        return shap_values

    def value(self, obs) -> float:
        explain = self.explain(obs)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float, device=self.device)
        self.mask.update(obs)
        scores = self.mask.compute(explain)
        action = np.argmax(self.agent.predict(obs_tensor))
        return scores[action]

    def supports(self, m: VisualizationMethod) -> bool:
        if not isinstance(m, VisualizationMethod):
            m = VisualizationMethod(m)
        return m == VisualizationMethod.RGB_ARRAY

    def getVisualizationParamsType(self, m: VisualizationMethod) -> type | None:
        if not isinstance(m, VisualizationMethod):
            m = VisualizationMethod(m)
        if m == VisualizationMethod.RGB_ARRAY:
            return ShapVisualizationParams
        return None

    def getVisualization(self, m: VisualizationMethod, params: Any = None) -> np.ndarray | dict | None:
        if self.last_explain is None or self.last_obs is None:
            return np.zeros((84, 84, 3), dtype=np.float32)
        
        if not isinstance(m, VisualizationMethod):
            m = VisualizationMethod(m)
        
        if m == VisualizationMethod.RGB_ARRAY:
            # Get the last frame from observations
            if self.last_obs.ndim == 4:
                last_obs = self.last_obs[-1]
            elif self.last_obs.ndim == 3:
                last_obs = self.last_obs
            else:
                raise ValueError(f"Unexpected observation shape: {self.last_obs.shape}")
            
            # Extract the last frame (most recent grayscale frame)
            if last_obs.ndim == 3:
                last_frame = last_obs[-1]  # Get the most recent frame
            elif last_obs.ndim == 2:
                last_frame = last_obs
            else:
                raise ValueError(f"Unexpected frame shape: {last_obs.shape}")
            
            idx = self.last_action if params is None or params.mode == "Last Action" else params.action
            idx = max(0, idx) % self.mask.action_space

            # Get SHAP values for the specific action and last frame
            if isinstance(self.last_explain, list):
                # Multiple outputs (one per action)
                shap_vals = self.last_explain[idx]
            else:
                # Single output, select the action dimension
                shap_vals = self.last_explain[..., idx]
            
            # Extract heatmap for the last frame
            if shap_vals.ndim == 4:  # (batch, channels, height, width)
                heatmap = shap_vals[0, -1, :, :]  # Last channel (most recent frame)
            elif shap_vals.ndim == 3:  # (channels, height, width)
                heatmap = shap_vals[-1, :, :]  # Last channel
            elif shap_vals.ndim == 2:  # (height, width)
                heatmap = shap_vals
            else:
                # Fallback: average across all dimensions except spatial
                heatmap = np.mean(shap_vals, axis=tuple(range(shap_vals.ndim - 2)))

            # Convert grayscale frame to RGB
            obs_img = gray2rgb(last_frame)
            obs_img = obs_img / 255.0 if obs_img.max() > 1.0 else obs_img

            # get the higher magnitude "min or max"
            scale = np.max(np.abs(heatmap))
            
            # Normalize heatmap
            heatmap_norm = (heatmap + scale) / (2 * scale + 1e-8) # Normalize to [0, 1]
            
            # Create masks for positive (red) and negative (blue) attributions
            red_mask = heatmap_norm >= 1 - params.threshold
            blue_mask = heatmap_norm <= params.threshold
            important = red_mask | blue_mask

            # Create colored overlay
            colored = np.zeros_like(obs_img)
            colored[red_mask, 0] = heatmap_norm[red_mask]  # Red for positive
            colored[blue_mask, 2] = 1 - heatmap_norm[blue_mask]  # Blue for negative

            # Create alpha channel for blending
            alpha = np.zeros((obs_img.shape[0], obs_img.shape[1], 1), dtype=np.float32)
            alpha[important] = 0.5

            # Blend the original image with the colored heatmap
            blended = (1 - alpha) * obs_img + alpha * colored
            return np.clip(blended, 0, 1)
        
        return None