from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from pearl.agent import RLAgent
from pearl.env import RLEnvironment
from pearl.mask import Mask
from pearl.method import ExplainabilityMethod


class LinearModelUTreeExplainability(ExplainabilityMethod):
    def __init__(self, device, mask: Mask):
        super().__init__()
        self.device = device
        self.explainer = None
        self.mask = mask
        self.agents = None
        self.trees = None
        self.linear_models = None
        self.training_data = []
        self.training_targets = []
        self.original_shape = None

    def set(self, env: RLEnvironment):
        super().set(env)
        obs = env.get_observations()
        self.original_shape = obs.shape  # Store original shape for later reshaping
        self.input_dim = np.prod(obs.shape[1:])  # Total number of features

    def prepare(self, agents: list[RLAgent]):
        self.agents = agents
        self.trees = []
        self.linear_models = []
        
        for _ in agents:
            # Initialize a decision tree for feature selection
            tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5)
            # Initialize a linear model for each leaf node
            linear_model = LinearRegression()
            self.trees.append(tree)
            self.linear_models.append(linear_model)

    def onStep(self, action: Any):
        # nothing for LMUT
        pass

    def onStepAfter(self, action: Any):
        # nothing for LMUT
        pass

    def add_training_data(self, obs, q_values):
        """Add training data for fitting the models"""
        self.training_data.append(obs)
        self.training_targets.append(q_values)

    def fit_models(self):
        """Fit the decision trees and linear models with collected data"""
        if not self.training_data or not self.training_targets:
            raise ValueError("No training data available. Call add_training_data first.")

        # Convert training data to 2D array
        X = np.vstack(self.training_data)  # Stack all observations into a 2D array
        y = np.vstack(self.training_targets)  # Stack all targets into a 2D array

        for i, (tree, linear_model) in enumerate(zip(self.trees, self.linear_models)):
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

    def explain(self, obs) -> np.ndarray | Any:
        if self.trees is None or self.linear_models is None:
            raise ValueError("Explainer not set. Please call prepare() first.")

        try:
            # Check if models are fitted using scikit-learn's validation
            for tree in self.trees:
                check_is_fitted(tree)
        except Exception as e:
            raise ValueError("Models not fitted. Call fit_models() first.") from e

        obs_tensor = torch.as_tensor(obs, dtype=torch.float, device=self.device)
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
                reshaped_importance = combined_importance.reshape(obs_shape)
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
                    reshaped_importance = combined_importance.reshape(obs_shape)
                    expanded_importance = np.expand_dims(reshaped_importance, axis=-1)
                    expanded_importance = np.repeat(expanded_importance, n_actions, axis=-1)
                except:
                    # Fallback: create a simple 5D tensor
                    reshaped_importance = combined_importance.reshape(-1)
                    expanded_importance = np.zeros((1, len(reshaped_importance), 1, 1, n_actions))
                    for a in range(n_actions):
                        expanded_importance[0, :, 0, 0, a] = reshaped_importance
            
            result.append(expanded_importance)
            
        return result

    def value(self, obs) -> list[float]:
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
    
    

    def visualize_tree_structure(self, agent_idx=0, save_path=None, show_plot=True, feature_names=None):
        """
        Visualize the decision tree structure.
        
        Args:
            agent_idx: Index of the agent to visualize (default: 0)
            save_path: Path to save the visualization (optional)
            show_plot: Whether to display the plot (default: True)
            feature_names: List of feature names to display instead of x[i] (optional)
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
        
        # Create tree visualization with much wider figure and better spacing
        fig, ax = plt.subplots(1, 1, figsize=(35, 20))  # Even wider and taller
        
        # Use sklearn's tree plotting with parameters to prevent collapsing
        plot_tree(tree, ax=ax, filled=True, rounded=True, fontsize=10, 
                 feature_names=feature_names, class_names=None, precision=3,
                 proportion=True, max_depth=None)
        ax.set_title(f'Decision Tree Structure for Agent {agent_idx}', fontsize=18, pad=25)
        
        # Adjust layout to prevent node overlapping
        plt.tight_layout(pad=2.0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig

    def visualize_feature_importance_bar(self, agent_idx=0, top_k=20, save_path=None, show_plot=True, feature_names=None):
        """
        Visualize feature importance as a bar chart.
        
        Args:
            agent_idx: Index of the agent to visualize (default: 0)
            top_k: Number of top features to display (default: 20)
            save_path: Path to save the visualization (optional)
            show_plot: Whether to display the plot (default: True)
            feature_names: List of feature names to display (optional)
        """
        if self.trees is None or self.linear_models is None:
            raise ValueError("Explainer not set. Please call prepare() first.")

        try:
            check_is_fitted(self.trees[agent_idx])
        except Exception as e:
            raise ValueError("Models not fitted. Call fit_models() first.") from e

        tree = self.trees[agent_idx]
        linear_model = self.linear_models[agent_idx]
        
        # Get feature importances
        tree_importance = tree.feature_importances_
        linear_importance = linear_model.coef_ if hasattr(linear_model, 'coef_') else np.zeros(self.input_dim)
        combined_importance = (tree_importance + linear_importance) / 2
        
        # Auto-generate feature names if not provided
        if feature_names is None:
            obs_shape = self.original_shape
            if len(obs_shape) == 4:  # Image-based environment
                channels, height, width = obs_shape[1], obs_shape[2], obs_shape[3]
                feature_names = []
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            feature_names.append(f"ch{c}_pos_{h}x{w}")
            elif len(obs_shape) == 2:  # Tabular environment
                num_features = obs_shape[1]
                feature_names = [f"feature_{i}" for i in range(num_features)]
            else:
                num_features = np.prod(obs_shape[1:])
                feature_names = [f"feature_{i}" for i in range(num_features)]
        
        # Get top k features
        top_indices = np.argsort(combined_importance)[-top_k:][::-1]
        top_importance = combined_importance[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        # Create the plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Combined importance
        bars1 = ax1.barh(range(len(top_names)), top_importance, color='skyblue', alpha=0.7)
        ax1.set_yticks(range(len(top_names)))
        ax1.set_yticklabels(top_names, fontsize=10)
        ax1.set_xlabel('Combined Feature Importance', fontsize=12)
        ax1.set_title(f'Top {top_k} Features - Combined Importance (Agent {agent_idx})', fontsize=14)
        ax1.invert_yaxis()
        
        # Tree importance
        tree_top_importance = tree_importance[top_indices]
        bars2 = ax2.barh(range(len(top_names)), tree_top_importance, color='lightgreen', alpha=0.7)
        ax2.set_yticks(range(len(top_names)))
        ax2.set_yticklabels(top_names, fontsize=10)
        ax2.set_xlabel('Tree Feature Importance', fontsize=12)
        ax2.set_title(f'Top {top_k} Features - Tree Importance (Agent {agent_idx})', fontsize=14)
        ax2.invert_yaxis()
        
        # Linear importance
        linear_top_importance = linear_importance[top_indices]
        bars3 = ax3.barh(range(len(top_names)), linear_top_importance, color='salmon', alpha=0.7)
        ax3.set_yticks(range(len(top_names)))
        ax3.set_yticklabels(top_names, fontsize=10)
        ax3.set_xlabel('Linear Model Coefficients', fontsize=12)
        ax3.set_title(f'Top {top_k} Features - Linear Coefficients (Agent {agent_idx})', fontsize=14)
        ax3.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig

    def visualize_feature_importance_heatmap(self, agent_idx=0, save_path=None, show_plot=True, cmap='viridis'):
        """
        Visualize feature importance as a heatmap (especially useful for image-based environments).
        
        Args:
            agent_idx: Index of the agent to visualize (default: 0)
            save_path: Path to save the visualization (optional)
            show_plot: Whether to display the plot (default: True)
            cmap: Colormap to use (default: 'viridis')
        """
        if self.trees is None or self.linear_models is None:
            raise ValueError("Explainer not set. Please call prepare() first.")

        try:
            check_is_fitted(self.trees[agent_idx])
        except Exception as e:
            raise ValueError("Models not fitted. Call fit_models() first.") from e

        tree = self.trees[agent_idx]
        linear_model = self.linear_models[agent_idx]
        
        # Get feature importances
        tree_importance = tree.feature_importances_
        linear_importance = linear_model.coef_ if hasattr(linear_model, 'coef_') else np.zeros(self.input_dim)
        combined_importance = (tree_importance + linear_importance) / 2
        
        obs_shape = self.original_shape
        
        if len(obs_shape) == 4:  # Image-based environment
            # Shape: (batch, channels, height, width)
            channels, height, width = obs_shape[1], obs_shape[2], obs_shape[3]
            
            # Create subplots for each channel
            fig, axes = plt.subplots(2, channels, figsize=(5*channels, 8))
            if channels == 1:
                axes = axes.reshape(2, 1)
            
            for c in range(channels):
                # Extract importance for this channel
                start_idx = c * height * width
                end_idx = (c + 1) * height * width
                channel_importance = combined_importance[start_idx:end_idx].reshape(height, width)
                
                # Combined importance heatmap
                im1 = axes[0, c].imshow(channel_importance, cmap=cmap, aspect='auto')
                axes[0, c].set_title(f'Channel {c} - Combined Importance', fontsize=12)
                axes[0, c].set_xlabel('Width')
                axes[0, c].set_ylabel('Height')
                plt.colorbar(im1, ax=axes[0, c])
                
                # Tree importance heatmap
                tree_channel_importance = tree_importance[start_idx:end_idx].reshape(height, width)
                im2 = axes[1, c].imshow(tree_channel_importance, cmap=cmap, aspect='auto')
                axes[1, c].set_title(f'Channel {c} - Tree Importance', fontsize=12)
                axes[1, c].set_xlabel('Width')
                axes[1, c].set_ylabel('Height')
                plt.colorbar(im2, ax=axes[1, c])
            
            plt.suptitle(f'Feature Importance Heatmaps for Agent {agent_idx}', fontsize=16)
            
        elif len(obs_shape) == 2:  # Tabular environment
            # Create a simple 1D heatmap
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8))
            
            # Combined importance
            im1 = ax1.imshow(combined_importance.reshape(1, -1), cmap=cmap, aspect='auto')
            ax1.set_title(f'Combined Feature Importance (Agent {agent_idx})', fontsize=14)
            ax1.set_xlabel('Feature Index')
            ax1.set_yticks([])
            plt.colorbar(im1, ax=ax1)
            
            # Tree importance
            im2 = ax2.imshow(tree_importance.reshape(1, -1), cmap=cmap, aspect='auto')
            ax2.set_title(f'Tree Feature Importance (Agent {agent_idx})', fontsize=14)
            ax2.set_xlabel('Feature Index')
            ax2.set_yticks([])
            plt.colorbar(im2, ax=ax2)
            
            # Linear importance
            im3 = ax3.imshow(linear_importance.reshape(1, -1), cmap=cmap, aspect='auto')
            ax3.set_title(f'Linear Model Coefficients (Agent {agent_idx})', fontsize=14)
            ax3.set_xlabel('Feature Index')
            ax3.set_yticks([])
            plt.colorbar(im3, ax=ax3)
            
        else:
            # Generic case
            fig, ax = plt.subplots(1, 1, figsize=(15, 4))
            im = ax.imshow(combined_importance.reshape(1, -1), cmap=cmap, aspect='auto')
            ax.set_title(f'Feature Importance Heatmap (Agent {agent_idx})', fontsize=14)
            ax.set_xlabel('Feature Index')
            ax.set_yticks([])
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig

    def visualize_feature_importance_comparison(self, save_path=None, show_plot=True, top_k=10, feature_names=None):
        """
        Compare feature importance across all agents.
        
        Args:
            save_path: Path to save the visualization (optional)
            show_plot: Whether to display the plot (default: True)
            top_k: Number of top features to display (default: 10)
            feature_names: List of feature names to display (optional)
        """
        if self.trees is None or self.linear_models is None:
            raise ValueError("Explainer not set. Please call prepare() first.")

        # Check if all models are fitted
        for i, tree in enumerate(self.trees):
            try:
                check_is_fitted(tree)
            except Exception as e:
                raise ValueError(f"Model for agent {i} not fitted. Call fit_models() first.") from e

        # Get feature importances for all agents
        all_importances = []
        for i, (tree, linear_model) in enumerate(zip(self.trees, self.linear_models)):
            tree_importance = tree.feature_importances_
            linear_importance = linear_model.coef_ if hasattr(linear_model, 'coef_') else np.zeros(self.input_dim)
            combined_importance = (tree_importance + linear_importance) / 2
            all_importances.append(combined_importance)
        
        all_importances = np.array(all_importances)  # Shape: (n_agents, n_features)
        
        # Get top k features based on average importance across agents
        avg_importance = np.mean(all_importances, axis=0)
        top_indices = np.argsort(avg_importance)[-top_k:][::-1]
        
        # Auto-generate feature names if not provided
        if feature_names is None:
            obs_shape = self.original_shape
            if len(obs_shape) == 4:  # Image-based environment
                channels, height, width = obs_shape[1], obs_shape[2], obs_shape[3]
                feature_names = []
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            feature_names.append(f"ch{c}_pos_{h}x{w}")
            elif len(obs_shape) == 2:  # Tabular environment
                num_features = obs_shape[1]
                feature_names = [f"feature_{i}" for i in range(num_features)]
            else:
                num_features = np.prod(obs_shape[1:])
                feature_names = [f"feature_{i}" for i in range(num_features)]
        
        # Create the comparison plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Bar chart comparing agents
        x = np.arange(len(top_indices))
        width = 0.8 / len(self.agents)
        
        for i, agent_importance in enumerate(all_importances):
            top_agent_importance = agent_importance[top_indices]
            ax1.bar(x + i * width, top_agent_importance, width, 
                   label=f'Agent {i}', alpha=0.7)
        
        ax1.set_xlabel('Feature')
        ax1.set_ylabel('Feature Importance')
        ax1.set_title(f'Top {top_k} Features - Comparison Across Agents', fontsize=14)
        ax1.set_xticks(x + width * (len(self.agents) - 1) / 2)
        # Use feature names if available, otherwise use indices
        if feature_names and len(feature_names) > max(top_indices):
            top_feature_names = [feature_names[idx] for idx in top_indices]
            ax1.set_xticklabels(top_feature_names, rotation=45, ha='right')
        else:
            ax1.set_xticklabels([f'F{idx}' for idx in top_indices])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Heatmap of all agents and top features
        top_features_importance = all_importances[:, top_indices]
        im = ax2.imshow(top_features_importance, cmap='viridis', aspect='auto')
        ax2.set_xlabel('Feature')
        ax2.set_ylabel('Agent Index')
        ax2.set_title(f'Feature Importance Heatmap - Top {top_k} Features', fontsize=14)
        ax2.set_xticks(range(len(top_indices)))
        # Use feature names if available, otherwise use indices
        if feature_names and len(feature_names) > max(top_indices):
            top_feature_names = [feature_names[idx] for idx in top_indices]
            ax2.set_xticklabels(top_feature_names, rotation=45, ha='right')
        else:
            ax2.set_xticklabels([f'F{idx}' for idx in top_indices])
        ax2.set_yticks(range(len(self.agents)))
        ax2.set_yticklabels([f'Agent {i}' for i in range(len(self.agents))])
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig

    def visualize_feature_importance_summary(self, save_path=None, show_plot=True):
        """
        Create a comprehensive summary of feature importance across all agents.
        
        Args:
            save_path: Path to save the visualization (optional)
            show_plot: Whether to display the plot (default: True)
        """
        if self.trees is None or self.linear_models is None:
            raise ValueError("Explainer not set. Please call prepare() first.")

        # Check if all models are fitted
        for i, tree in enumerate(self.trees):
            try:
                check_is_fitted(tree)
            except Exception as e:
                raise ValueError(f"Model for agent {i} not fitted. Call fit_models() first.") from e

        # Get feature importances for all agents
        all_importances = []
        tree_importances = []
        linear_importances = []
        
        for i, (tree, linear_model) in enumerate(zip(self.trees, self.linear_models)):
            tree_importance = tree.feature_importances_
            linear_importance = linear_model.coef_ if hasattr(linear_model, 'coef_') else np.zeros(self.input_dim)
            combined_importance = (tree_importance + linear_importance) / 2
            
            all_importances.append(combined_importance)
            tree_importances.append(tree_importance)
            linear_importances.append(linear_importance)
        
        all_importances = np.array(all_importances)
        tree_importances = np.array(tree_importances)
        linear_importances = np.array(linear_importances)
        
        # Create summary statistics
        mean_importance = np.mean(all_importances, axis=0)
        std_importance = np.std(all_importances, axis=0)
        max_importance = np.max(all_importances, axis=0)
        min_importance = np.min(all_importances, axis=0)
        
        # Create the summary plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Mean importance with error bars
        feature_indices = np.arange(len(mean_importance))
        ax1.errorbar(feature_indices, mean_importance, yerr=std_importance, 
                    fmt='o-', alpha=0.7, capsize=3)
        ax1.set_xlabel('Feature Index')
        ax1.set_ylabel('Mean Feature Importance')
        ax1.set_title('Mean Feature Importance Across Agents', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Box plot of importance distribution
        ax2.boxplot(all_importances.T, labels=[f'Agent {i}' for i in range(len(self.agents))])
        ax2.set_xlabel('Agent')
        ax2.set_ylabel('Feature Importance')
        ax2.set_title('Feature Importance Distribution by Agent', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Max vs Min importance
        ax3.scatter(min_importance, max_importance, alpha=0.6)
        ax3.plot([min_importance.min(), min_importance.max()], 
                [min_importance.min(), min_importance.max()], 'r--', alpha=0.5)
        ax3.set_xlabel('Minimum Importance')
        ax3.set_ylabel('Maximum Importance')
        ax3.set_title('Max vs Min Feature Importance', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Tree vs Linear importance correlation
        mean_tree = np.mean(tree_importances, axis=0)
        mean_linear = np.mean(linear_importances, axis=0)
        ax4.scatter(mean_tree, mean_linear, alpha=0.6)
        ax4.set_xlabel('Mean Tree Importance')
        ax4.set_ylabel('Mean Linear Importance')
        ax4.set_title('Tree vs Linear Model Importance', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Feature Importance Summary Across All Agents', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig

    def collect_training_data(self, num_samples: int = 1000):
        """Collect training data by running the environment and gathering Q-values from agents.
        
        Args:
            num_samples (int): Number of training samples to collect. Defaults to 1000.
        """
        from tqdm import tqdm
        
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
            obs_reshaped = obs.reshape(obs.shape[0], -1)  # This will give us (batch_size, features)
            self.add_training_data(obs_reshaped, q_values)
            
            # Take a random action to get new observations
            action = self.env.action_space.sample()
            state, reward_dict, terminated, truncated, info = self.env.step(action)
            if terminated:
                self.env.reset()