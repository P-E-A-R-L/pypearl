import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import cv2
import ale_py

# Register Atari environments
gym.register_envs(ale_py)

# Hyperparameters
ENV_NAME       = "ALE/Assault-v5"
MEMORY_SIZE    = 100_000
BATCH_SIZE     = 32
GAMMA          = 0.99
LEARNING_RATE  = 1e-4
SYNC_TARGET    = 10_000
EPS_START      = 0.1
EPS_END        = 0.1
EPS_DECAY      = 1_000_000
MAX_FRAMES     = 1_000
FRAMES_PRINT   = 100000
FRAME_SKIP     = 4
STACK_SIZE     = 4
MODEL_PATH     = "models/dqn_assault_1k.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing wrappers
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        from gymnasium.spaces import Box
        self.observation_space = Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=2)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        from gymnasium.spaces import Box
        self.k = k
        self.frames = collections.deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = Box(
            low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs)
        stacked = np.concatenate(self.frames, axis=2)
        return stacked, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        stacked = np.concatenate(self.frames, axis=2)
        return stacked, reward, terminated, truncated, info

def make_env(seed=None):
    env = gym.make(ENV_NAME, obs_type='rgb', frameskip=FRAME_SKIP)
    if seed is not None:
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
    env = PreprocessFrame(env)
    env = FrameStack(env, STACK_SIZE)
    return env

# Q-network
class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(STACK_SIZE, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        # x shape: N,H,W,C -> N,C,H,W
        # print(x.shape)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        return self.net(x)

# Replay buffer
Transition = collections.namedtuple('Transition',
                                    ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

def compute_loss(batch, policy_net, target_net):
    states      = torch.tensor(np.stack(batch.state), dtype=torch.uint8, device=DEVICE)
    next_states = torch.tensor(np.stack(batch.next_state), dtype=torch.uint8, device=DEVICE)
    actions     = torch.tensor(batch.action, dtype=torch.int64, device=DEVICE).unsqueeze(1)
    rewards     = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    dones       = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0].unsqueeze(1)
        target = rewards + GAMMA * next_q * (1 - dones)
    return torch.nn.functional.mse_loss(q_values, target)


env = make_env(seed=1)
n_actions = env.action_space.n

policy_net = DQN(n_actions).to(DEVICE)
target_net = DQN(n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
buffer = ReplayBuffer(MEMORY_SIZE)

state, info = env.reset()
epsilon = EPS_START
frame_idx = 0
episode_reward = 0



while frame_idx < MAX_FRAMES:
    epsilon = max(EPS_END, EPS_START - frame_idx / EPS_DECAY)

    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.uint8, device=DEVICE).unsqueeze(0)
            q_vals = policy_net(state_t)
            action = q_vals.argmax(1).item()

    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    buffer.push(state, action, reward, next_state, done)
    state = next_state
    episode_reward += reward
    frame_idx += 1

    if len(buffer) >= BATCH_SIZE:
        batch = buffer.sample(BATCH_SIZE)
        loss = compute_loss(batch, policy_net, target_net)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if frame_idx % SYNC_TARGET == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if frame_idx % FRAMES_PRINT == 0:
         print(f"Frame: {frame_idx}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.4f}")

    if done:
        # print(f"Frame: {frame_idx}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.4f}")
        state, info = env.reset()
        episode_reward = 0

torch.save(policy_net.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

env.close()

if __name__ == "__main__":
    pass