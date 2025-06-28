from typing import Any, List
import numpy as np
import torch
import shap

from pearl.agent import RLAgent
from pearl.env import RLEnvironment
from pearl.mask import Mask
from pearl.method import ExplainabilityMethod

from pearl.lab.visual import VisualizationMethod
from pearl.lab.annotations import Param

class TabularShapVisualizationParams:
    mode: Param(str, choices=["Last Action", "Selected Action"]) = "Last Action"
    action: Param(int) = 0


class TabularShapExplainability(ExplainabilityMethod):
    def __init__(self, device: torch.device, mask: Mask, feature_names: List[str]):
        super().__init__()
        self.device = device
        self.mask = mask
        self.feature_names = feature_names if isinstance(feature_names, list) else feature_names.split(',')
        self.agent: RLAgent = None
        self.explainer = None
        self.background_size = 100
        self.last_explain = None

    def set(self, env: RLEnvironment):
        super().set(env)
        self.env = env
        self.background = np.array(
            [env.observation_space.sample() for _ in range(self.background_size)],
            dtype=np.float32
        ).reshape(self.background_size, -1)

    def prepare(self, agent: RLAgent):
        self.agent = agent
        model = agent.get_q_net().to(self.device).eval()

        def predict_fn(x: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
                logits = model(x_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        self.explainer = shap.KernelExplainer(predict_fn, self.background)

    def onStep(self, action):
        self.last_action = action
        
    def onStepAfter(self, action, reward, done, info): pass

    def explain(self, obs: np.ndarray) -> Any:
        obs_vec = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        shap_vals = self.explainer.shap_values(obs_vec, silent=True).squeeze().T

        explanation = {
            'shap_values': np.array(shap_vals),  # shape: (action, features)
            'feature_names': self.feature_names,
            'data': obs_vec.squeeze()
        }
        self.last_explain = explanation
        return explanation

    def value(self, obs: np.ndarray) -> float:
        exp = self.explain(obs)
        self.mask.update(obs)

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).squeeze()
        with torch.no_grad():
            q_vals = self.agent.get_q_net()(obs_tensor)

        action = int(torch.argmax(q_vals))
        shap_values = exp['shap_values']
        weights = shap_values.T

        score = float(self.mask.compute(weights)[action])
        action_q = q_vals[action].item()
        max_q = torch.max(q_vals).item()
        confidence = action_q / max_q if max_q != 0 else 1.0
        return score * confidence

    def supports(self, m: VisualizationMethod) -> bool:
        return VisualizationMethod(m) == VisualizationMethod.BAR_CHART

    def getVisualizationParamsType(self, m: VisualizationMethod) -> type | None:
        if VisualizationMethod(m) == VisualizationMethod.BAR_CHART:
            return TabularShapVisualizationParams
        return None

    def getVisualization(self, m: VisualizationMethod, params: Any = None) -> dict | None:
        if VisualizationMethod(m) != VisualizationMethod.BAR_CHART:
            return None
        if self.last_explain is None:
            return {name: 0.0 for name in self.feature_names}
        
        if params.mode == "Last Action":
                idx = self.last_action
        else:
            idx = 0
            if params is not None and isinstance(params, TabularShapVisualizationParams):
                idx = params.action
            idx = max(0, idx) % self.mask.action_space
        
        vals = np.array(self.last_explain['shap_values'])[idx]
        return {self.feature_names[i]: float(vals[i]) for i in range(len(self.feature_names))}
