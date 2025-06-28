from dowhy import gcm
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import gym
from reinforce import Agent

# Sample function to collect agent data for causal analysis
def collect_trajectory_data(agent, env, num_episodes=100):
    data = []
    for _ in range(num_episodes):
        observation, _ = env.reset()
        done = False
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            data.append({
                "x_position": observation[0],
                "y_position": observation[1],
                "x_velocity": observation[2],
                "y_velocity": observation[3],
                "angle": observation[4],
                "angular_velocity": observation[5],
                "left_leg_contact": observation[6],
                "right_leg_contact": observation[7],
                "action": action,
                "reward": reward
            })
            observation = next_observation
    return pd.DataFrame(data)

def create_causal_model(data):
    causal_model = gcm.StructuralCausalModel(nx.DiGraph())
    
    # Define causal relationships (assumed based on domain knowledge)
    causal_model.graph.add_edges_from([
        ("x_position", "action"),
        ("y_position", "action"),
        ("x_velocity", "action"),
        ("y_velocity", "action"),
        ("angle", "action"),
        ("angular_velocity", "action"),
        ("left_leg_contact", "action"),
        ("right_leg_contact", "action"),
        ("action", "reward")
    ])
    
    # Assign causal mechanisms
    for node in causal_model.graph.nodes:
        if node == "action":
            # Use a conditional stochastic model for the action node
            causal_model.set_causal_mechanism(node, gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
        elif node == "reward":
            # Use a regression model for the reward node
            causal_model.set_causal_mechanism(node, gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
        else:
            # Use empirical distribution for nodes without parents
            causal_model.set_causal_mechanism(node, gcm.EmpiricalDistribution())
    
    # Fit the causal mechanisms to the data
    gcm.auto.assign_causal_mechanisms(causal_model, data)
    
    return causal_model

# Visualize causal graph
def plot_causal_graph(causal_model):
    plt.figure(figsize=(8, 6))
    nx.draw(causal_model.graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10)
    plt.title("Causal Graph of Agent's Decision-Making")
    plt.show()

# Perform intervention analysis
def perform_intervention(causal_model, data, intervention_var, new_value):
    gcm.auto.assign_causal_mechanisms(causal_model, data)
    intervened_data = gcm.interventional_samples(
        causal_model, interventions={intervention_var: new_value}, num_samples=100
    )
    return intervened_data

# Example usage:
env = gym.make('LunarLander-v2')
agent = Agent(alpha=0.0005, gamma=0.99, n_actions=4, input_dims=env.observation_space.shape[0])
agent.load_model("lunar_lander_reinforce_1000.pth")
df = collect_trajectory_data(agent, env, num_episodes=10)
model = create_causal_model(df)
plot_causal_graph(model)
#new_data = perform_intervention(model, df, "y_velocity", 0.5)