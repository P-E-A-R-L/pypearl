from typing import Any
import numpy as np
import torch
from skimage.color import gray2rgb
from skimage.transform import resize

from pearl.agent import RLAgent
from pearl.env import RLEnvironment
from pearl.mask import Mask
from pearl.method import ExplainabilityMethod
from pearl.custom_methods.customLimeImage import CustomLimeImageExplainer

from pearl.lab.visual import VisualizationMethod
from pearl.lab.annotations import Param

class LimeVisualizationParams:
    mode: Param(str, choices=["Last Action", "Selected Action"]) = "Last Action"
    action: Param(int) = 0
    threshold: Param(float, choices=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]) = 0.4

class LimeExplainability(ExplainabilityMethod):
    """
    LIME explainability aligned with the new ShapExplainability interface.
    Uses only the most recent frame from a stack of grayscale frames.
    """
    def __init__(self, device: torch.device, mask: Mask):
        super().__init__()
        self.device = device
        self.mask = mask
        self.explainer = CustomLimeImageExplainer(num_samples=500, num_segments=500)
        self.agent: RLAgent = None
        self.last_explain = None
        self.obs = None

    def set(self, env: RLEnvironment):
        super().set(env)

    def prepare(self, agent: RLAgent):
        self.agent = agent

    def onStep(self, action: Any):
        self.last_action = action

    def onStepAfter(self, action: Any, reward: dict, done: bool, info: dict):
        pass

    def explain(self, obs: np.ndarray) -> Any:
        if self.agent is None:
            raise ValueError("Call prepare() before explain().")

        self.obs = obs
        frame = obs.squeeze()  # (C, H, W) or (H, W)
        # Determine channel count for Q-net input
        orig_C = frame.shape[0] if frame.ndim == 3 else 1
        # Use only the most recent grayscale frame
        if frame.ndim == 3:
            gray_frame = frame[-1, :, :]
        elif frame.ndim == 2:
            gray_frame = frame
        else:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")
        # Convert to RGB for LIME
        img = gray2rgb(gray_frame)

        def batch_predict(images: np.ndarray) -> np.ndarray:
            gr = np.mean(images, axis=3, keepdims=True).astype(np.float32) / 255.0
            stacked = np.repeat(gr, orig_C, axis=3)
            tensor = torch.tensor(
                np.transpose(stacked, (0, 3, 1, 2)), dtype=torch.float32
            ).to(self.device)
            with torch.no_grad():
                out = self.agent.get_q_net()(tensor)
            return out.cpu().numpy()

        exp = self.explainer.explain_instance(
            image=img.astype(np.double),
            classifier_fn=batch_predict,
            top_labels=self.mask.action_space,
            hide_color=0,
            num_samples=100,
        )

        self.last_explain = exp
        self.last_obs = obs
        return exp

    def value(self, obs: np.ndarray) -> float:
        exp = self.explain(obs)
        self.mask.update(obs)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.agent.get_q_net()(obs_t)
        action = int(torch.argmax(q))

        segs = exp.segments
        A, C, H, W = self.mask.action_space, *obs.shape[1:]

        maps = np.zeros((A, H, W), dtype=float)
        for lbl, pairs in exp.local_exp.items():
            for seg_id, wt in pairs:
                maps[lbl, segs == seg_id] = wt

        if segs.shape != (H, W):
            maps = np.stack([
                resize(m, (H, W), preserve_range=True, anti_aliasing=True)
                for m in maps
            ], axis=0)

        maps_hw_a = np.transpose(maps, (1, 2, 0))
        attributions = np.abs(maps_hw_a[None, None, :, :, :].astype(np.float32))
        attributions /= np.sum(attributions, axis=(1, 2, 3), keepdims=True)

        return float(self.mask.compute(attributions)[action])

    def supports(self, m: VisualizationMethod) -> bool:
        if not isinstance(m, VisualizationMethod):
            m = VisualizationMethod(m)
        return m == VisualizationMethod.RGB_ARRAY

    def getVisualizationParamsType(self, m: VisualizationMethod) -> type | None:
        if not isinstance(m, VisualizationMethod):
            m = VisualizationMethod(m)
        return LimeVisualizationParams if m == VisualizationMethod.RGB_ARRAY else None

    def getVisualization(self, m: VisualizationMethod, params: Any = None) -> np.ndarray | dict | None:
        if self.last_explain is None or self.last_obs is None:
            return np.zeros((84, 84, 3), dtype=np.float32)
        if not isinstance(m, VisualizationMethod):
            m = VisualizationMethod(m)
        if m == VisualizationMethod.RGB_ARRAY:
            segs = self.last_explain.segments
            heatmap = np.zeros(segs.shape, dtype=np.float32)
            idx = self.last_action if params is None or params.mode == "Last Action" else params.action
            idx = max(0, idx) % self.mask.action_space
            
            for segment, importance in self.last_explain.local_exp[idx]:
                heatmap[segs == segment] = importance

            if self.last_obs.ndim == 4:
                self.last_obs = self.last_obs[-1]
            if self.last_obs.ndim == 3:
                last_frame = self.last_obs[-1]
            elif self.last_obs.ndim == 2:
                last_frame = self.last_obs
            else:
                raise ValueError(f"Unexpected observation shape: {self.last_obs.shape}")

            obs_img = gray2rgb(last_frame)
            obs_img = obs_img / 255.0 if obs_img.max() > 1.0 else obs_img

            scale = np.max(np.abs(heatmap))
            heatmap_norm = (heatmap + scale) / (2 * scale + 1e-8)
            red_mask = heatmap_norm >= 1 - params.threshold
            blue_mask = heatmap_norm <= params.threshold
            important = red_mask | blue_mask

            colored = np.zeros_like(obs_img)
            colored[red_mask, 0] = heatmap_norm[red_mask]
            colored[blue_mask, 2] = 1 - heatmap_norm[blue_mask]

            alpha = np.zeros((obs_img.shape[0], obs_img.shape[1], 1), dtype=np.float32)
            alpha[important] = 0.5

            blended = (1 - alpha) * obs_img + alpha * colored
            return np.clip(blended, 0, 1)
        return None
