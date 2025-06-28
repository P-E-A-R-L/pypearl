import torch
import numpy as np
from pearl.agent import RLAgent
from stable_baselines3 import DQN

class DQNAgent(RLAgent):
    """
    Simple DQN agent.
    """
    def __init__(self, model_path: str):
        self.m_agent = DQN.load(model_path)
        self.q_net = self.m_agent.q_net

    def predict(self, observation):
        self.q_net.eval()
        with torch.no_grad():
            if isinstance(observation, np.ndarray):
                observation = torch.tensor(observation).to(next(self.q_net.parameters()).device,
                                                           next(self.q_net.parameters()).dtype)
            values = self.q_net(observation)
            exp_q = np.exp(values)
            return exp_q / np.sum(exp_q, axis=1, keepdims=True)

    def get_q_net(self):
        return self.q_net

    def close(self):
        del self.m_agent