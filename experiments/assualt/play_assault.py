import os
import time
import numpy as np
import torch
import torch.nn as nn
import cv2
import gymnasium as gym

from pearl.agents.TourchDQN import TorchDQN
from pearl.enviroments.ObservationWrapper import ObservationWrapper
from pearl.enviroments.GymRLEnv import GymRLEnv

# Constants
ENV_NAME       = "ALE/Assault-v5"
FRAME_SKIP     = 4
STACK_SIZE     = 4
MODEL_PATH     = "models/dqn_assault_5m.pth"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISPLAY_WIDTH   = 400
DISPLAY_HEIGHT  = 300

class DQN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(STACK_SIZE, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),      nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),      nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7*7*64, 512),                          nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, H, W, C) -> (N, C, H, W)
        # print(x.shape)
        # x = x.permute(0, 3, 1, 2).float() / 255.0
        return self.net(x)


def preprocess(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized

class Wrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        state_t = np.expand_dims(obs, axis=0)
        state_t = np.transpose(state_t, (0, 3, 1, 2))
        state_t = state_t.astype(np.double) / 255.0
        return state_t

    def get_observations(self):
        return self.observation(self.env.get_observations())

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward_dict, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward_dict, terminated, truncated, info


def save_image_batch(batch: np.ndarray, out_dir: str, prefix: str = "img"):
    """
    Save a batch of images of shape (N, C, H, W) to disk with OpenCV.

    Args:
      batch (np.ndarray): array of images, dtype=uint8, shape (N, H, W, C)
      out_dir (str): directory to save into (will be created if it doesn't exist)
      prefix (str): filename prefix; saved files will be prefix_0.png, prefix_1.png, ...
    """
    os.makedirs(out_dir, exist_ok=True)
    print(batch.shape)

    N = batch.shape[1]
    for i in range(N):
        img = batch[0,i,:,:]                   # shape (H, W, C)
        filename = os.path.join(out_dir, f"{prefix}_{i}.png")
        # OpenCV expects BGR for color; if your array is RGB, convert:
        print(img.shape)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_to_save = img
        cv2.imwrite(filename, img_to_save)

    print(f"Saved {N} images to '{out_dir}/'")

def play(
    num_episodes: int = 5,
    display_size: tuple = (DISPLAY_WIDTH, DISPLAY_HEIGHT)
) -> None:
    # Initialize our wrapped env with rgb_array rendering
    env = Wrapper(GymRLEnv(
        env_name=ENV_NAME,
        stack_size=STACK_SIZE,
        frame_skip=FRAME_SKIP,
        render_mode='rgb_array',
        observation_preprocessing=preprocess,
    ))
    n_actions = env.action_space.n

    policy_net = DQN(n_actions)
    agent = TorchDQN(MODEL_PATH, policy_net, DEVICE)

    # Setup display window
    cv2.namedWindow('Assault (small)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Assault (small)', *display_size)

    run = False

    for ep in range(1, num_episodes + 1):
        state, info = env.reset()
        done = False
        total_reward = 0.0
        run = False
        step = False
        while not done:
            # render frame array
            frame = env.render(mode='rgb_array')
            small = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)
            cv2.imshow('Assault (small)', small)
            time.sleep(1/20.0)
            key = cv2.waitKey(1) & 0xFF
            if  key == ord('q'):
                done = True
                break

            if key == ord('n'):
                step = True

            if key == ord('s'):
                save_image_batch(state * 255, ".")

            if key == ord('r'):
                run = not run

            if step:
                run = True

            if not run:
                continue

            if step:
                step = False
                run = False

            # select action
            state_t = torch.tensor(state, dtype=torch.float, device=DEVICE)
            action = agent.predict(state_t)

            state, reward_dict, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward_dict['reward'])

        print(f"Episode {ep} — Reward: {total_reward:.2f}")

    env.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    play(num_episodes=5)
