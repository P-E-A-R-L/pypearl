import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

os.makedirs("./graphs", exist_ok=True)

counter = 0

class PolicyNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        return self.softmax(logits)

def train_agent(num_episodes, spurious_acc_sampler, true_acc, training_states_bias='true', lr=1e-2, device='cpu'):
    policy = PolicyNet().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    rewards = []

    for episode in range(1, num_episodes + 1):
        true_feats = np.random.choice([-1, 1], size=3)
        spurious_feat = np.random.choice([-1, 1])
        sum_true = np.sum(true_feats)
        combo_sign = 1 if sum_true >= 0 else -1
        spurious_acc = spurious_acc_sampler()

        obs = np.array([*true_feats, spurious_feat], dtype=np.float32)
        obs_tensor = torch.from_numpy(obs).float().to(device)

        probs = policy(obs_tensor)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        action_val = -1 if action.item() == 0 else 1
        
        reward = 0.0
        # This is the true combination reward and the one used later for evaluation
        if np.random.rand() < true_acc:
            reward = 1.0 if action_val == combo_sign else reward
            
        # For the spurious bias agent,     
        if np.random.rand() < spurious_acc:
            reward = 1.0 if action_val == spurious_feat else reward

        loss = -m.log_prob(action) * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards.append(reward)

        if episode % 2000 == 0:
            avg_r = np.mean(rewards[-2000:])
            print(f"[{training_states_bias.upper()} Agent] Episode {episode}, Avg Reward (last 2000): {avg_r:.3f}")

    return policy

def evaluate_policy_true(policy, num_samples=1000, device='cpu'):
    total = 0
    for _ in range(num_samples):
        true_feats = np.random.choice([-1, 1], size=3)
        spurious_feat = np.random.choice([-1, 1])
        sum_true = np.sum(true_feats)
        combo_sign = 1 if sum_true >= 0 else -1

        obs = np.array([*true_feats, spurious_feat], dtype=np.float32)
        obs_tensor = torch.from_numpy(obs).float().to(device)

        with torch.no_grad():
            probs = policy(obs_tensor)
        action = torch.argmax(probs).item()
        action_val = -1 if action == 0 else 1

        total += (action_val == combo_sign)

    return total / num_samples

# --- Parameters ---
num_episodes = 10000
lr = 1e-2
device = 'cpu'

# The spurious accuracy samplers define the probability of the spurious feature leading to a correct action.
# we use it to simulate the second agent that hacks the environment by focusing on the spurious feature.
spurious_acc_sampler_true  = lambda: np.random.uniform(0.0, 0.2)
spurious_acc_sampler_spurious = lambda: np.random.uniform(0.9, 1.0)
# The true accuracy is the probability of the true combination leading to a correct action.
true_acc = 0.8

# Train agents
policy1 = train_agent(num_episodes, spurious_acc_sampler_true, true_acc, training_states_bias='true', lr=lr, device=device)
policy2 = train_agent(num_episodes, spurious_acc_sampler_spurious, true_acc, training_states_bias='spurious', lr=lr, device=device)

# Evaluate under test correlations
results = []
for _ in range(5):
    r1 = evaluate_policy_true(policy1, num_samples=2000, device=device)
    r2 = evaluate_policy_true(policy2, num_samples=2000, device=device)
    results.append((r1, r2))

print("\nEvaluation Results (avg reward wrt true combination):")
for  index, (r1, r2) in enumerate(results, start=1):
    print(f"Test {index:.1f}: Agent1 avg reward={r1:.3f}, Agent2 avg reward={r2:.3f}")

# --- LIME Explainability ---
try:
    from lime.lime_tabular import LimeTabularExplainer

    background_data = np.random.uniform(-1, 1, size=(500, 4))
    explainer = LimeTabularExplainer(
        background_data,
        feature_names=['true1', 'true2', 'true3', 'spurious'],
        discretize_continuous=False,
        mode='classification'
    )

    def make_predict_fn(policy):
        def predict_fn(x):
            with torch.no_grad():
                x_tensor = torch.from_numpy(x.astype(np.float32)).float().to(device)
                probs = policy(x_tensor).cpu().numpy()
            return probs
        return predict_fn

    predict_fn1 = make_predict_fn(policy1)
    predict_fn2 = make_predict_fn(policy2)

    combos = []
    for t_bits in range(8):
        true_feats = [1 if (t_bits >> i) & 1 else -1 for i in range(3)]
        for s in [-1, 1]:
            combos.append(true_feats + [s])
    combos = np.array(combos, dtype=np.float32)

    weights1 = []
    weights2 = []
    for obs in combos:
        exp1 = explainer.explain_instance(obs, predict_fn1, num_features=4)
        exp2 = explainer.explain_instance(obs, predict_fn2, num_features=4)

        def parse_weights(exp_list, feat):
            for cond, weight in exp_list:
                if feat in cond:
                    return weight
            return 0.0

        w1 = [parse_weights(exp1.as_list(), fn) for fn in ['true1', 'true2', 'true3', 'spurious']]
        w2 = [parse_weights(exp2.as_list(), fn) for fn in ['true1', 'true2', 'true3', 'spurious']]
        weights1.append(w1)
        weights2.append(w2)

    weights1 = np.array(weights1)
    weights2 = np.array(weights2)

    avg_abs1 = np.mean(np.abs(weights1), axis=0)
    avg_abs2 = np.mean(np.abs(weights2), axis=0)

    fig, ax = plt.subplots()
    x = np.arange(4)
    width = 0.35
    ax.bar(x - width/2, avg_abs1, width, label='Agent1 (true-primary)')
    ax.bar(x + width/2, avg_abs2, width, label='Agent2 (spurious-primary)')
    ax.set_xticks(x)
    ax.set_xticklabels(['true1', 'true2', 'true3', 'spurious'])
    ax.set_ylabel('Average |LIME weight|')
    ax.set_title('Average Feature Importance by LIME')
    ax.legend()
    plt.tight_layout()
    plt.savefig('./graphs/avg_feature_importance.png')
    plt.close(fig)

    def plot_heatmap(weights, title, filename):
        fig, ax = plt.subplots(figsize=(6, 8))
        im = ax.imshow(weights, aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('Features')
        ax.set_ylabel('Combo index')
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(['t1', 't2', 't3', 'spur'])
        fig.colorbar(im, ax=ax, orientation='vertical', label='LIME weight')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    plot_heatmap(weights1, 'Agent1 LIME weights per combo', './graphs/agent1_weights_heatmap.png')
    plot_heatmap(weights2, 'Agent2 LIME weights per combo', './graphs/agent2_weights_heatmap.png')

    print("\nSaved plots:")
    print(" - ./graphs/avg_feature_importance.png")
    print(" - ./graphs/agent1_weights_heatmap.png")
    print(" - ./graphs/agent2_weights_heatmap.png")

except ImportError:
    print("\nLIME not installed. Install it with: pip install lime")
