from abc import ABC, abstractmethod
import numpy as np
from typing import Any


class RLAgent(ABC):
    """
    Simple abstract base class for Reinforcement Learning agents.
    """

    @abstractmethod
    def __init__(self, observation_space=None, action_space=None):
        """
        Initialize the agent.

        Args:
            observation_space: The observation space of the environment
            action_space: The action space of the environment
        """
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def predict(self, observation: Any) -> np.ndarray:
        """
        Returns the probability for each action given an observation.

        Args:
            observation: The current observation from the environment

        Returns:
            action_probs: Probability distribution over actions
        """
        pass

    @abstractmethod
    def get_q_net(self) -> Any:
        """
        Returns the Q-network of the agent.

        Returns:
            q_net: The Q-network of the agent
        """
        pass

    def close(self) -> None:
        """
        Clean up resources used by the agent.
        """
        pass