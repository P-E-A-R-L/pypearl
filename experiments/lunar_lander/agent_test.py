import gym
from reinforce import Agent
import numpy as np
import matplotlib.pyplot as plt
import os

actions = {0: "Do Nothing", 1: "Left Engine", 2: "Main Engine", 3: "Right Engine"}
counter = 0

class RewardDecompositionTracker:
    def __init__(self, gamma=0.99):
        self.rewards = []  # Stores tuples of decomposed rewards
        self.gamma = gamma  # Discount factor
    
    def store(self, decomposed_rewards):
        """Store decomposed rewards for later return calculation."""
        self.rewards.append(decomposed_rewards)
    
    def compute_returns(self):
        """Compute discounted returns for each component."""
        n = len(self.rewards)
        G = {key: np.zeros(n) for key in self.rewards[0]}  # Initialize returns

        # Compute returns for each reward component
        for t in reversed(range(n)):
            for key in G:
                G[key][t] = self.rewards[t][key] + (self.gamma * G[key][t+1] if t+1 < n else 0)
        
        return G  # Returns a dictionary of component-wise returns

def decomposed_reward(state, action, terminated, truncated, reward):
    x, y, vx, vy, angle, ang_vel, leg1_contact, leg2_contact = state
    global counter
    # Decomposed reward components
    distance_penalty = -0.3 * abs(x)  
    angle_penalty = -0.2 * abs(angle)  
    time_penalty = -0.1
    main_engine_cost = -0.03 if action == 2 else 0  
    side_engine_cost = -0.015 if action in [1, 3] else 0  
    leg_contact = 10 * (leg1_contact + leg2_contact)  
    crash_penalty = -100 if terminated else 0  
    success_bonus = 100 if truncated else 0  
 
    return {
        "distance_penalty": distance_penalty,
        "angle_penalty": angle_penalty,
        "time_penalty": time_penalty,
        "main_engine_cost": main_engine_cost,
        "side_engine_cost": side_engine_cost,
        "leg_contact": leg_contact,
        "crash_penalty": crash_penalty,
        "success_bonus": success_bonus
    }
    
def is_correct_placement(state):
    x, y, vx, vy, angle, ang_vel, leg1_contact, leg2_contact = state
    correct_placement = (abs(x) < 0.02 and abs(y) < 0.02 and abs(vy) < 0.05 and leg1_contact and leg2_contact)
    return correct_placement


def try_one_round():
    env = gym.make('LunarLander-v2', render_mode="human")
    agent = Agent(alpha=0.0005, gamma=0.99, n_actions=4, input_dims=env.observation_space.shape[0])
    agent.load_model("lunar_lander_reinforce.pth")

    score = 0
    observation, info = env.reset()
    done = False

    reward_tracker = RewardDecompositionTracker(gamma=0.99)
    action_history = []
    
    while not done:
        action, action_probs = agent.choose_action(observation, return_probs=True)
        observation_, reward, terminated, truncated, info = env.step(action)
        
        # if the placement is correct we terminate early
        # if is_correct_placement(observation_):
        #     terminated = True
        #     reward = 100
        
        # Store decomposed rewards
        decomposed = decomposed_reward(observation, action, terminated, truncated, reward)
        reward_tracker.store(decomposed)
        
        action_history.append((observation, action, decomposed))
        
        done = terminated or truncated
        agent.store_transition(observation, action, reward)
        score += reward
        observation = observation_
        env.render()
        # wait for a click inside the window
        #plt.waitforbuttonpress()
    
    env.close()
    print(f"Score: {score}")

    # Compute decomposed Q-values (discounted returns)
    decomposed_q_values = reward_tracker.compute_returns()
    
    # Explain actions using Reward Difference Explanation (RDX)
    plot_rdx(action_history, decomposed_q_values)


def plot_rdx(action_history, decomposed_q_values, save_folder="reward_decomposition"):
    """Generates 4 graphs—one per action—showing how reward components influenced each action."""
    
    # Create folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Initialize a dictionary to store aggregated differences per action
    aggregated_diff = {action: {key: 0 for key in decomposed_q_values} for action in actions}

    # Count occurrences of each action to compute averages
    action_counts = {action: 0 for action in actions}

    # Sum contributions per action
    num_states = len(action_history) - 1  # Number of transitions
    for t in range(num_states):
        action = action_history[t][1]  # Get action at time t
        action_counts[action] += 1  # Increment action count

        for key in decomposed_q_values:
            aggregated_diff[action][key] += decomposed_q_values[key][t] - decomposed_q_values[key][t + 1]

    # Convert sums to averages
    for action in actions:
        if action_counts[action] > 0:
            for key in aggregated_diff[action]:
                aggregated_diff[action][key] /= action_counts[action]

    # Plot and save a graph for each action
    for action in actions:
        plt.figure(figsize=(8, 8))
        plt.bar(aggregated_diff[action].keys(), aggregated_diff[action].values(), 
                color=['g' if v > 0 else 'r' for v in aggregated_diff[action].values()])
        plt.xlabel("Reward Component")
        plt.ylabel("Average Contribution to Action Selection")
        plt.title(f"Reward Decomposition for Action: {actions[action]}")
        plt.xticks(rotation=20)
        plt.grid()

        save_path = os.path.join(save_folder, f"{actions[action].replace(' ', '_')}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Saved {save_path}")

if __name__ == "__main__":
    try_one_round()
