from typing import Any, List
import numpy as np
import torch

from pearl.agent import RLAgent
from pearl.env import RLEnvironment
from pearl.mask import Mask
from pearl.method import ExplainabilityMethod
from lime.lime_tabular import LimeTabularExplainer

from pearl.lab.visual import VisualizationMethod
from pearl.lab.annotations import Param

class TabularLimeVisualizationParams:
    mode: Param(str, choices=["Last Action", "Selected Action"]) = "Last Action"
    action: Param(int) = 0

class TabularLimeExplainability(ExplainabilityMethod):
    def __init__(self, device: torch.device, mask: Mask, feature_names: List[str]):
        super().__init__()
        self.device = device
        self.mask = mask
        if isinstance(feature_names, str):
            self.feature_names = feature_names.split(",")
        else:
            self.feature_names = feature_names
        self.agent: RLAgent = None

        self.explainer = LimeTabularExplainer(
            training_data=np.zeros((1, len(self.feature_names))),
            feature_names=self.feature_names,
            mode="classification",
            discretize_continuous=False
        )
        self.last_explain = None

    def set(self, env: RLEnvironment):
        super().set(env)

    def prepare(self, agent: RLAgent):
        self.agent = agent

    def onStep(self, action: Any): 
        self.last_action = action
    def onStepAfter(self, action: Any, reward: dict, done: bool, info: dict): pass

    def explain(self, obs: np.ndarray) -> Any:
        if self.agent is None:
            raise ValueError("Call prepare() before explain().")

        obs_vec = obs.squeeze()
        if obs_vec.ndim == 2:
            obs_vec = obs_vec[0]

        model = self.agent.get_q_net().to(self.device).eval()

        def predict_fn(x: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
                logits = model(x_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        obs_tensor = torch.tensor(obs_vec.reshape(1, -1), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_vals = model(obs_tensor)
        num_actions = q_vals.shape[1]

        exp = self.explainer.explain_instance(
            data_row=obs_vec,
            predict_fn=predict_fn,
            num_features=len(self.feature_names),
            top_labels=num_actions,
            num_samples=1000,
        )

        self.last_explain = exp
        return exp

    def value(self, obs: np.ndarray) -> float:
        exp = self.explain(obs)
        self.mask.update(obs)

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).squeeze()
        with torch.no_grad():
            q_vals = self.agent.get_q_net()(obs_tensor)

        action = int(torch.argmax(q_vals))

        weights = np.zeros((len(self.feature_names), self.mask.action_space), dtype=np.float32)
        for a in range(self.mask.action_space):
            for fid, weight in exp.local_exp.get(a, []):
                weights[fid, a] = weight

        score = float(self.mask.compute(weights)[action])
        action_q = q_vals[action].item()
        max_q = torch.max(q_vals).item()
        confidence = action_q / max_q if max_q != 0 else 1.0

        return score * confidence

    def supports(self, m: VisualizationMethod) -> bool:
        if not isinstance(m, VisualizationMethod):
            m = VisualizationMethod(m)
        return m == VisualizationMethod.BAR_CHART

    def getVisualizationParamsType(self, m: VisualizationMethod) -> type | None:
        if not isinstance(m, VisualizationMethod):
            m = VisualizationMethod(m)
        if m == VisualizationMethod.BAR_CHART:
            return TabularLimeVisualizationParams
        return None

    def getVisualization(self, m: VisualizationMethod, params: Any = None) -> dict | None:
        if not isinstance(m, VisualizationMethod):
            m = VisualizationMethod(m)
        if m == VisualizationMethod.BAR_CHART:
            if self.last_explain is None:
                return {name: 0.0 for name in self.feature_names}
            
            if params.mode == "Last Action":
                idx = self.last_action
            else:
                idx = 0
                if params is not None and isinstance(params, TabularLimeVisualizationParams):
                    idx = params.action
                idx = max(0, idx) % self.mask.action_space
            
            return {self.feature_names[fid]: weight for fid, weight in self.last_explain.local_exp.get(idx, [])}
        return None
