from typing import Any, List, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from pearl.agent import RLAgent
from pearl.env import RLEnvironment
from pearl.mask import Mask
from pearl.method import ExplainabilityMethod

from pearl.lab.visual import VisualizationMethod
from pearl.lab.annotations import Param

class LMUTVisualizationParams:
    action: Param(int) = 0
    agent_idx: Param(int) = 0


class LinearModelUTreeExplainability(ExplainabilityMethod):
    """
    Linear Model U-Tree (LMUT) explainability method.
    
    Combines decision trees for feature selection with linear models for each leaf node
    to provide interpretable explanations of agent behavior.
    """
    
    def __init__(self, device: torch.device, mask: Mask, 
                 max_depth: int = 5, min_samples_leaf: int = 5,
                 num_training_samples: int = 1000):
        """
        Initialize LMUT explainability method.
        
        Args:
            device: PyTorch device for computations
            mask: Mask object for computing attribution scores
            max_depth: Maximum depth of decision trees
            min_samples_leaf: Minimum samples per leaf node
            num_training_samples: Number of training samples to collect
        """
        super().__init__()
        self.device = device
        self.mask = mask
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.num_training_samples = num_training_samples
        
        # Core components
        self.agents: Optional[List[RLAgent]] = None
        self.trees: Optional[List[DecisionTreeRegressor]] = None
        self.linear_models: Optional[List[LinearRegression]] = None
        
        # Training data
        self.training_data: List[np.ndarray] = []
        self.training_targets: List[np.ndarray] = []
        self.original_shape: Optional[tuple] = None
        self.input_dim: Optional[int] = None
        
        # Visualization state
        self.last_explain: Optional[List[np.ndarray]] = None
        self.last_obs: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None

    def set(self, env: RLEnvironment):
        """Set the environment and initialize shape information."""
        super().set(env)
        obs = env.get_observations()
        self.original_shape = obs.shape
        self.input_dim = np.prod(obs.shape[1:])  # Total number of features

    def prepare(self, agents: RLAgent):
        """Prepare the explainer with agents and initialize models."""
        # Handle single agent case
        if not isinstance(agents, list):
            agents = [agents]
        
        self.agents = agents
        self.trees = []
        self.linear_models = []
        
        for _ in agents:
            # Initialize a decision tree for feature selection
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, 
                min_samples_leaf=self.min_samples_leaf
            )
            # Initialize a linear model for each leaf node
            linear_model = LinearRegression()
            self.trees.append(tree)
            self.linear_models.append(linear_model)

    def onStep(self, action: Any):
        """Called before step - store the action for later use."""
        self.last_action = action

    def onStepAfter(self, action: Any, reward: dict, done: bool, info: dict):
        """Called after step - no action needed for LMUT."""
        pass

    def add_training_data(self, obs: np.ndarray, q_values: np.ndarray):
        """Add training data for fitting the models."""
        self.training_data.append(obs)
        self.training_targets.append(q_values)

    def collect_training_data(self, num_samples: Optional[int] = None):
        """Collect training data by running the environment and gathering Q-values from agents."""
        if num_samples is None:
            num_samples = self.num_training_samples
            
        if self.agents is None:
            raise ValueError("Agents not set. Call prepare() first.")
        
        print(f"Collecting {num_samples} training samples...")
        for _ in tqdm(range(num_samples), desc="Collecting training samples"):
            obs = self.env.get_observations()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float, device=self.device)
            
            # Get Q-values from all agents
            q_values = []
            for agent in self.agents:
                with torch.no_grad():
                    q_vals = agent.get_q_net()(obs_tensor).cpu().numpy()
                    # Take the maximum Q-value for each agent
                    q_values.append(np.max(q_vals, axis=1))
            q_values = np.array(q_values).T  # Shape: (batch_size, n_agents)
            
            # Reshape observation to 2D array (flatten all dimensions except batch)
            obs_reshaped = obs.reshape(obs.shape[0], -1)  # (batch_size, features)
            self.add_training_data(obs_reshaped, q_values)
            
            # Take a random action to get new observations
            action = self.env.action_space.sample()
            state, reward_dict, terminated, truncated, info = self.env.step(action)
            if terminated:
                self.env.reset()

    def fit_models(self):
        """Fit the decision trees and linear models with collected data."""
        if not self.training_data or not self.training_targets:
            raise ValueError("No training data available. Call collect_training_data() first.")

        if self.trees is None or self.linear_models is None:
            raise ValueError("Models not initialized. Call prepare() first.")

        print("Fitting LMUT models...")
        
        # Convert training data to 2D array
        X = np.vstack(self.training_data)  # Stack all observations into a 2D array
        y = np.vstack(self.training_targets)  # Stack all targets into a 2D array

        for i, (tree, linear_model) in enumerate(zip(self.trees, self.linear_models)):
            print(f"Fitting models for agent {i}...")
            
            # Fit the decision tree
            tree.fit(X, y[:, i])
            
            # Get the leaf nodes for each sample
            leaf_indices = tree.apply(X)
            
            # Fit linear models for each leaf node
            unique_leaves = np.unique(leaf_indices)
            for leaf in unique_leaves:
                mask = leaf_indices == leaf
                if np.sum(mask) > 1:  # Only fit if we have enough samples
                    linear_model.fit(X[mask], y[mask, i])

    def explain(self, obs: np.ndarray) -> List[np.ndarray]:
        """Generate explanations using the fitted LMUT models."""
        if self.trees is None or self.linear_models is None:
            raise ValueError("Explainer not set. Please call prepare() first.")

        try:
            # Check if models are fitted using scikit-learn's validation
            for tree in self.trees:
                check_is_fitted(tree)
        except Exception as e:
            raise ValueError("Models not fitted. Call fit_models() first.") from e

        result = []
        
        for i, (tree, linear_model) in enumerate(zip(self.trees, self.linear_models)):
            # Get feature importance from the tree
            tree_importance = tree.feature_importances_
            # Get linear model coefficients
            linear_importance = linear_model.coef_ if hasattr(linear_model, 'coef_') else np.zeros(self.input_dim)
            # Combine both importances
            combined_importance = (tree_importance + linear_importance) / 2
            
            # Get the number of actions from the environment
            n_actions = self.env.action_space.n
            
            # Detect environment type based on observation shape
            obs_shape = self.original_shape
            
            if len(obs_shape) == 4:  # Image-based environment (e.g., Space Invaders)
                # Shape: (batch, channels, height, width)
                # Reshape to original image shape
                reshaped_importance = combined_importance.reshape(obs_shape[1:])
                # Expand to include action channels
                expanded_importance = np.expand_dims(reshaped_importance, axis=-1)
                expanded_importance = np.repeat(expanded_importance, n_actions, axis=-1)
                
            elif len(obs_shape) == 2:  # Tabular environment (e.g., Lunar Lander)
                # Shape: (batch, features)
                # Remove batch dimension and create 5D tensor for mask compatibility
                reshaped_importance = combined_importance.reshape(obs_shape[1:])  # Remove batch dimension
                # Create the expected 5D tensor shape: (1, features, 1, 1, actions)
                expanded_importance = np.expand_dims(reshaped_importance, axis=0)  # Add batch dim
                expanded_importance = np.expand_dims(expanded_importance, axis=2)  # Add height dim
                expanded_importance = np.expand_dims(expanded_importance, axis=3)  # Add width dim
                expanded_importance = np.expand_dims(expanded_importance, axis=4)  # Add action dim
                expanded_importance = np.repeat(expanded_importance, n_actions, axis=4)  # Repeat for all actions
                
            else:
                # Handle other cases (1D, 3D, etc.)
                # Try to reshape to original shape and add action dimension
                try:
                    reshaped_importance = combined_importance.reshape(obs_shape[1:])
                    expanded_importance = np.expand_dims(reshaped_importance, axis=-1)
                    expanded_importance = np.repeat(expanded_importance, n_actions, axis=-1)
                except:
                    # Fallback: create a simple 5D tensor
                    reshaped_importance = combined_importance.reshape(-1)
                    expanded_importance = np.zeros((1, len(reshaped_importance), 1, 1, n_actions))
                    for a in range(n_actions):
                        expanded_importance[0, :, 0, 0, a] = reshaped_importance
            
            result.append(expanded_importance)
        
        # Store for visualization
        self.last_explain = result
        self.last_obs = obs
        
        return result

    def value(self, obs: np.ndarray) -> List[float]:
        """Compute attribution values for the current observation."""
        explains = self.explain(obs)
        values = []
        
        for i, explain in enumerate(explains):
            agent = self.agents[i]
            # Reshape observation back to original shape for the mask
            obs_original = obs.reshape(self.original_shape)
            self.mask.update(obs_original)
            scores = self.mask.compute(explain)
            # Convert to tensor and use original shape for DQN prediction
            obs_tensor = torch.as_tensor(obs_original, dtype=torch.float, device=self.device)
            action = agent.predict(obs_tensor)
            values.append(scores[action])
            
        return values

    def supports(self, m: VisualizationMethod) -> bool:
        """Check if the method supports the given visualization type."""
        if not isinstance(m, VisualizationMethod):
            m = VisualizationMethod(m)
        return m == VisualizationMethod.RGB_ARRAY

    def getVisualizationParamsType(self, m: VisualizationMethod) -> type | None:
        """Get the visualization parameters type for the given method."""
        if not isinstance(m, VisualizationMethod):
            m = VisualizationMethod(m)
        return LMUTVisualizationParams if m == VisualizationMethod.RGB_ARRAY else None

    def getVisualization(self, m: VisualizationMethod, params: Any = None) -> np.ndarray | dict | None:
        """Generate visualization for the given method."""
        if self.last_explain is None or self.last_obs is None:
            return np.zeros((84, 84, 3), dtype=np.float32)
        
        if not isinstance(m, VisualizationMethod):
            m = VisualizationMethod(m)
        
        if m == VisualizationMethod.RGB_ARRAY:
            # Get parameters
            if params is None:
                params = LMUTVisualizationParams()
            
            agent_idx = getattr(params, 'agent_idx', 0)
            action = getattr(params, 'action', self.last_action or 0)
            
            if agent_idx >= len(self.last_explain):
                return np.zeros((84, 84, 3), dtype=np.float32)
            
            # Get the explanation for the specified agent
            explain = self.last_explain[agent_idx]
            
            # Extract the heatmap for the specified action
            if explain.ndim == 5:  # (batch, channels, height, width, actions)
                heatmap = explain[0, :, :, :, action]  # Remove batch and action dims
            elif explain.ndim == 4:  # (channels, height, width, actions)
                heatmap = explain[:, :, :, action]
            elif explain.ndim == 3:  # (height, width, actions)
                heatmap = explain[:, :, action]
            else:
                # Fallback: average across all dimensions except spatial
                heatmap = np.mean(explain, axis=tuple(range(explain.ndim - 2)))
            
            # Convert to 2D heatmap if needed
            if heatmap.ndim > 2:
                heatmap = np.mean(heatmap, axis=0)  # Average across channels
            
            # Normalize heatmap
            scale = np.max(np.abs(heatmap)) + 1e-8
            heatmap_norm = (heatmap + scale) / (2 * scale)
            
            # Create RGB visualization
            obs_img = np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.float32)
            
            # Create masks for positive (red) and negative (blue) attributions
            red_mask = heatmap_norm >= 0.6
            blue_mask = heatmap_norm <= 0.4
            important = red_mask | blue_mask
            
            # Create colored overlay
            colored = np.zeros_like(obs_img)
            colored[red_mask, 0] = heatmap_norm[red_mask]  # Red for positive
            colored[blue_mask, 2] = 1 - heatmap_norm[blue_mask]  # Blue for negative
            
            # Create alpha channel for blending
            alpha = np.zeros((obs_img.shape[0], obs_img.shape[1], 1), dtype=np.float32)
            alpha[important] = 0.5
            
            # Blend the original image with the colored heatmap
            blended = (1 - alpha) * obs_img + alpha * colored
            return np.clip(blended, 0, 1)
        
        return None

    def visualize_tree_structure(self, agent_idx: int = 0, save_path: Optional[str] = None, 
                                show_plot: bool = True, feature_names: Optional[List[str]] = None) -> plt.Figure:
        """
        Visualize the decision tree structure.
        
        Args:
            agent_idx: Index of the agent to visualize (default: 0)
            save_path: Path to save the visualization (optional)
            show_plot: Whether to display the plot (default: True)
            feature_names: List of feature names to display instead of x[i] (optional)
            
        Returns:
            matplotlib Figure object
        """
        if self.trees is None:
            raise ValueError("Explainer not set. Please call prepare() first.")

        try:
            check_is_fitted(self.trees[agent_idx])
        except Exception as e:
            raise ValueError("Models not fitted. Call fit_models() first.") from e

        tree = self.trees[agent_idx]
        
        # Auto-generate feature names if not provided
        if feature_names is None:
            obs_shape = self.original_shape
            if len(obs_shape) == 4:  # Image-based environment
                # Shape: (batch, channels, height, width)
                channels, height, width = obs_shape[1], obs_shape[2], obs_shape[3]
                feature_names = []
                
                # Create meaningful names for image pixels
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            # Use channel and position information
                            feature_names.append(f"ch{c}_pos_{h}x{w}")
                                
            elif len(obs_shape) == 2:  # Tabular environment
                # Shape: (batch, features)
                num_features = obs_shape[1]
                feature_names = [f"feature_{i}" for i in range(num_features)]
            else:
                # Generic naming for other cases
                num_features = np.prod(obs_shape[1:])
                feature_names = [f"feature_{i}" for i in range(num_features)]
        
        # Create tree visualization with wide figure and better spacing
        fig, ax = plt.subplots(1, 1, figsize=(35, 20))
        
        # Use sklearn's tree plotting with parameters to prevent collapsing
        plot_tree(tree, ax=ax, filled=True, rounded=True, fontsize=10, 
                 feature_names=feature_names, class_names=None, precision=3,
                 proportion=True, max_depth=None)
        ax.set_title(f'LMUT Decision Tree Structure for Agent {agent_idx}', fontsize=18, pad=25)
        
        # Adjust layout to prevent node overlapping
        plt.tight_layout(pad=2.0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
