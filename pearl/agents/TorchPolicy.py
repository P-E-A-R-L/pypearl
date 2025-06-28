import torch
from pearl.agent import RLAgent

class TorchPolicyAgent(RLAgent):
    """
    A generic policy-gradient agent using PyTorch (e.g., REINFORCE).
    """

    def __init__(self, model_path: str, model_module, device):
        self.policy_net = model_module.to(device)
        self.policy_net.load_state_dict(torch.load(model_path, map_location=device))
        self.policy_net.eval()
        self.device = device

    def predict(self, observation):
        self.policy_net.eval()
        with torch.no_grad():
            if not isinstance(observation, torch.Tensor):
                observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
            if observation.ndim == 1:
                observation = observation.unsqueeze(0)
            logits = self.policy_net(observation)
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
        probs = probs.squeeze()                
        return probs.cpu().numpy()


    def get_q_net(self):
        return self.policy_net
        