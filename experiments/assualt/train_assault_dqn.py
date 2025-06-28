from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import os
import gc
import gymnasium as gym
import ale_py

# Register Atari environments
gym.register_envs(ale_py)

# Environment config
ENV_NAME = "ALE/Assault-v5"
N_ENVS = 12  # Increased from 4 to 8 for more parallel environments
FRAME_STACK = 4
SEED = 42  # Changed seed

# Model paths
LOG_DIR = "./improved_dqn_logs/"
MODEL_DIR = "./improved_models/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Improved CNN feature extractor with more capacity
class ImprovedCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        # Get shape of input channels
        n_input_channels = observation_space.shape[0]
        
        # Define a deeper CNN with more filters
        cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Get features dimension
        with torch.no_grad():
            sample_obs = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = cnn(sample_obs).shape[1]
            
        super().__init__(observation_space, features_dim=512)
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 512),  # First dense layer
            nn.ReLU(),
            nn.Linear(512, 512),  # Second dense layer
            nn.ReLU()
        )
        self.cnn = cnn
        

    def forward(self, observations):
        x = self.cnn(observations)
        x = self.fc(x)
        return x

# Create a separate evaluation environment
def make_eval_env():
    eval_env = make_atari_env(ENV_NAME, n_envs=1, seed=SEED+100)  # Different seed for eval
    eval_env = VecFrameStack(eval_env, n_stack=FRAME_STACK)
    eval_env = VecTransposeImage(eval_env)  # Properly transpose image for PyTorch
    return eval_env

# Create and wrap the training environment
env = make_atari_env(ENV_NAME, n_envs=N_ENVS, seed=SEED)
env = VecFrameStack(env, n_stack=FRAME_STACK)
env = VecTransposeImage(env)  # Properly transpose image for PyTorch

# Create evaluation environment
eval_env = make_eval_env()

# Configure the policy with our improved CNN
policy_kwargs = dict(
    features_extractor_class=ImprovedCNN,
    net_arch=[512, 256],  # Added explicit architecture for the value network
    optimizer_class=optim.Adam,
    optimizer_kwargs=dict(eps=1e-5),  # More stable Adam epsilon
)

# Create the improved DQN model with better hyperparameters
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=5e-5,  # Lower learning rate for more stability
    buffer_size=50000,  # Larger replay buffer
    learning_starts=100000,  # More exploration before learning
    batch_size=128,  # Larger batch size
    gamma=0.99,
    target_update_interval=10000,  # Less frequent target network updates
    exploration_fraction=0.4,
    exploration_final_eps=0.01,
    train_freq=4,
    gradient_steps=1,
    policy_kwargs=policy_kwargs,
    tensorboard_log=LOG_DIR,
    verbose=1,
    device='cuda' if torch.cuda.is_available() else 'cpu',  # Explicit device selection
)

# Evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{MODEL_DIR}/best/",
    log_path=f"{MODEL_DIR}/eval_logs/",
    eval_freq=50000,
    n_eval_episodes=10,
    deterministic=True,
    render=False
)

# GC and checkpoint callback
class EnhancedCallback(CheckpointCallback):
    def _on_step(self) -> bool:
        env.render("human")  # Render environment for better monitoring
        if self.n_calls % 250 == 0:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        return super()._on_step()

# Set up checkpoint callback
checkpoint_callback = EnhancedCallback(
    save_freq=100000,
    save_path=MODEL_DIR,
    name_prefix="dqn_spaceinvaders",
    verbose=1
)

# Set up callbacks
callbacks = [checkpoint_callback, eval_callback]

# Train the model for longer
model.learn(
    total_timesteps=1_000_000,  # Increased training time
    callback=callbacks,
    progress_bar=True,
    tb_log_name="improved_dqn_run"
)

# Save the final model
model.save(f"{MODEL_DIR}/dqn_final_improved")

# Close environments
env.close()
eval_env.close()

print("Training completed successfully!")