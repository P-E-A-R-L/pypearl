import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pearl.lab.annotations import Param
import torch.nn.functional as F


from pearl.provided.LunarLander import LunarLanderTabularMask
from pearl.methods.TabularShapExplainability import TabularShapExplainability
from pearl.methods.TabularLimeExplainability import TabularLimeExplainability
from pearl.agents.TorchPolicy import TorchPolicyAgent
from pearl.enviroments.GymRLEnv import GymRLEnv
from pearl.enviroments.ObservationWrapper import ObservationWrapper
from pearl.agents.TorchDQN import TorchDQN

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Correct network structure matching the trained REINFORCE model
class REINFORCE_Net(nn.Module):
    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.pi = nn.Linear(256, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.pi(x)  # Raw logits


class LunarWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, obs):
        # Convert to tensor for compatibility with the rest of the code
        return np.array(obs, dtype=np.float32)

    def get_observations(self):
        return self.observation(self.env.get_observations())

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = obs[:, 0]
        return self.observation(obs), reward, terminated, truncated, info


def cudaDevice():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


import pearl.Pearl as Pearl

if __name__ == "__main__":
    device = cudaDevice()

    env_factory = lambda agent: LunarWrapper(GymRLEnv(env_name='LunarLander-v3', tabular=True, stack_size=1))

    n_actions = 4
    input_dim = 8

    policy_net_good = DQN(input_dim, n_actions)
    agent_good = TorchPolicyAgent('experiments/lunar_lander/dqn_lunar_lander_agent48.pth', policy_net_good, device)

    policy_net_bad = REINFORCE_Net(input_dim, n_actions)
    agent_bad = TorchPolicyAgent('experiments/lunar_lander/lunar_lander_reinforce_2000.pth', policy_net_bad, device)

    agents = [agent_good, agent_bad]

    mask = LunarLanderTabularMask()
    feature_names = ["x_pos", "y_pos", "x_vel", "y_vel", "angle", "angular_vel", "leg1", "leg2"]

    method_1_factory = lambda agent: TabularLimeExplainability(device, mask, feature_names)

    print(Pearl.eval(
        agents,
        env_factory,
        [method_1_factory],
        [1],
        max_steps_ep=150,
    ))
