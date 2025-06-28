import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from pearl.visualizable import Visualizable


class RLEnvironment(Visualizable):
    """
    Abstract base class for Reinforcement Learning environments.

    This class defines the standard interface for RL environments
    following conventions similar to OpenAI Gym/Gymnasium.

    Child classes should manage any configuration parameters
    (e.g., frame stacking, reward processing) in their own
    constructors and not in this abstract base.
    """
    @abstractmethod
    def __init__(self):
        """
        Initialize the environment.

        Child classes should define:
        - observation_space: The space that defines valid observations
        - action_space: The space that defines valid actions
        - metadata: Dictionary with environment metadata (render modes, etc.)
        """
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Args:
            seed: Optional seed for reproducibility
            options: Additional options for resetting the environment

        Returns:
            observation: Initial observation (N, H, W, C)
            info: Additional information
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, Dict[str, np.ndarray], bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment with the given action.

        Args:
            action: The action to take (index of the action)

        Returns:
            observation: The next observation (N, H, W, C)
            reward: The reward for taking the action (decomposed reward)
            terminated: Whether the episode has terminated
            truncated: Whether the episode was truncated
            info: Additional information
        """
        pass

    @abstractmethod
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.

        Args:
            mode: The render mode (e.g., "human", "rgb_array")

        Returns:
            None if mode is "human" or an rgb array if mode is "rgb_array"
        """
        pass

    def close(self) -> None:
        """
        Clean up resources used by the environment.
        """
        pass

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Set the seed for the environment's random number generator.

        This method is included for backward compatibility. Modern implementations
        should use the seed parameter in reset().

        Args:
            seed: The seed value

        Returns:
            A list containing the seed
        """
        return [seed]

    @property
    def unwrapped(self):
        """
        Return the base non-wrapped environment.

        Returns:
            The base non-wrapped environment
        """
        if hasattr(self, 'env'):
            return self.env
        raise NotImplementedError(
            "unwrapped() is not implemented for this environment type."
        )

    def get_available_actions(self) -> List[Any]:
        """
        Get the list of available actions in the current state.

        Returns:
            A list of valid actions that can be taken in the current state
        """
        if hasattr(self.action_space, 'n'):
            return list(range(self.action_space.n))
        else:
            raise NotImplementedError(
                "get_available_actions() is not implemented for this action space type."
            )

    @abstractmethod
    def get_observations(self) -> Any:
        """
        Retrieve the current observations (e.g., stacked frames). (N, H, W, C)
        """
        pass