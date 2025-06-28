from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import numpy as np
import gymnasium as gym
import ale_py
import time

env_name = "ALE/SpaceInvaders-v5"
gym.register_envs(ale_py)

def test_model(model_path, num_episodes=10, render=True):
    # Create and wrap the environment
    env = make_atari_env(env_name, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)

    # Load the trained model
    model = DQN.load(model_path)

    episode_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Get model's action
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]  # Reward comes as array due to vectorized env

            if render:
                env.render("human")
                time.sleep(0.025)  # Add small delay to make rendering viewable

            if done:
                print(f"Episode {episode + 1} reward: {episode_reward}")
                episode_rewards.append(episode_reward)
                break

    env.close()

    # Print summary statistics
    print("\nTest Results:")
    print(f"Number of episodes: {num_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Standard deviation: {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")

    return episode_rewards


if __name__ == "__main__":
    model_path = "./models/best_model.zip"

    # Test the model
    rewards = test_model(
        model_path=model_path,
        num_episodes=10,
        render=True
    )
