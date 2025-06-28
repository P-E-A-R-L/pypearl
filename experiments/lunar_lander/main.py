import gym
import numpy as np
import matplotlib.pyplot as plt
from reinforce import Agent

def plotLearning(scores, filename, window=100):
    running_avg = np.zeros(len(scores))
    for i in range(len(scores)):
        running_avg[i] = np.mean(scores[max(0, i-window):(i+1)])
    plt.plot(running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(filename)
    plt.close()

def convert_to_serializable(obj):
    """
    Recursively convert NumPy types and other non-serializable objects to JSON-serializable types.
    """
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Convert dictionary keys to strings if they are not already
        return {str(key): convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        # Handle other types (e.g., custom objects) by converting them to strings
        return str(obj)

if __name__ == '__main__':
    env = gym.make('LunarLander-v2') #  render_mode="human"

    agent = Agent(alpha=0.0005, gamma=0.99, n_actions=4, input_dims=env.observation_space.shape[0])
    # Load a pre-trained model if needed "to continue training"
    # agent.load_model("lunar_lander_reinforce_1000.pth")
    score_history = []
    start_episode = 0
    num_episodes = 50000
    try:
        for i in range(num_episodes):
            done = False
            score = 0
            observation, info = env.reset()
            episode_explanations = []  # Store explanations for multiple steps
            episode_images = []  # Store image filenames
            episode_json_explanations = []  # Store JSON explanations
            # start json explanations with feature names
            episode_json_explanations.append({"feature_names": ["x_position", "y_position", "x_velocity", "y_velocity",
                        "angle", "angular_velocity", "left_leg_contact", "right_leg_contact"]})
            
            steps = 0
            while not done:
                action, action_probs = agent.choose_action(observation, return_probs=True)
                observation_, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                reward -= 0.4 * steps  # small time penalty
                agent.store_transition(observation, action, reward)
                score += reward
                observation = observation_

            score_history.append(score)
            agent.learn()
            avg_score = np.mean(score_history[-100:])
            print(f'episode: {i + start_episode}, score: {score:.1f}, average score: {avg_score:.1f}')
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final model...")
        agent.save_model("lunar_lander_reinforce_interrupted.pth")
        print("Final model saved. Exiting.")

    agent.save_model("lunar_lander_reinforce.pth")
    filename = 'lunar-lander.png'
    plotLearning(score_history, filename=filename, window=100)