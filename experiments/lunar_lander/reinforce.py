import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyGradientNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, fc1_dims=256, fc2_dims=256):
        super(PolicyGradientNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.pi(x), dim=-1)
        return action_probs

class Agent:
    def __init__(self, alpha=0.003, gamma=0.99, n_actions=4, input_dims=8, device=None):
        self.gamma = gamma
        self.n_actions = n_actions
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy = PolicyGradientNetwork(n_actions, input_dims).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=alpha)
        
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
    
    def choose_action(self, observation, return_probs=False):
        state = torch.tensor(observation, dtype=torch.float32).to(self.device)
        action_probs = self.policy(state)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()

        if return_probs:
            return action.item(), action_probs.cpu().detach().numpy()
        return action.item()

    
    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
    
    def learn(self):
        G = []
        rewards = self.reward_memory
        discounted_sum = 0
        for reward in reversed(rewards):
            discounted_sum = reward + (self.gamma * discounted_sum)
            G.insert(0, discounted_sum)
        
        G = torch.tensor(G, dtype=torch.float32).to(self.device)
        G = (G - G.mean()) / (G.std() + 1e-8)
        
        loss = 0
        for idx, (g, state) in enumerate(zip(G, self.state_memory)):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            action_probs = self.policy(state)
            action_dist = Categorical(action_probs)
            log_prob = action_dist.log_prob(torch.tensor(self.action_memory[idx], dtype=torch.float32).to(self.device))
            loss += -g * log_prob
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        
    def save_model(self, filename="reinforce_agent.pth"):
        torch.save(self.policy.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename="reinforce_agent.pth"):
        self.policy.load_state_dict(torch.load(filename, map_location=self.device))
        print(f"Model loaded from {filename}")

