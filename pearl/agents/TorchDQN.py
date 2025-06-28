import numpy as np

from pearl.agent import RLAgent
import torch

class TorchDQN(RLAgent):
    """
    Simple DQN agent using PyTorch.
    """
    def __init__(self, model_path: str, module, device):    
        self.m_agent = module.to(device)
        self.m_agent.load_state_dict(torch.load(model_path, map_location=device))
        self.m_agent.eval()
        self.q_net = self.m_agent.net if hasattr(self.m_agent, 'net') else self.m_agent.network if hasattr(self.m_agent, 'network') else self.m_agent
        self.device = device

    def predict(self, observation):
        self.q_net.eval()
        observation = torch.as_tensor(observation, dtype=torch.float, device=self.device)
        with torch.no_grad():
            q_vals = self.m_agent(observation)
            q_vals = q_vals.cpu().numpy() if q_vals.is_cuda else q_vals.numpy()
            q_vals = np.exp(q_vals - np.max(q_vals)) / np.sum(np.exp(q_vals - np.max(q_vals)))
            return q_vals.reshape(-1)

    def get_q_net(self):
        return self.q_net