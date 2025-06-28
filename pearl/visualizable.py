from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from pearl.lab.visual import VisualizationMethod
from pearl.lab.annotations import Param


class Visualizable(ABC):
    """
    Abstract base class for visualizable methods.

    This class defines the standard interface for methods that can be visualized.
    Child classes should implement the `visualize` method to provide their specific visualization logic.
    """

    @abstractmethod
    def supports(self, m: VisualizationMethod) -> bool:
        """
        Should return if this instance supports this type of visualization
        :param m: the visualization method to query
        :return: bool, whether this method is supported
        """
        return False

    @abstractmethod
    def getVisualizationParamsType(self, m: VisualizationMethod) -> type | None:
        """
        If the visualization method requires some parameters, then this function should return the type of the
        dataclass of the parameters the method expects.
        :param m: the visualization method to query
        :return: None if the method is not supported / requires no params, else the type of the dataclass
        """
        return None

    @abstractmethod
    def getVisualization(self, m: VisualizationMethod, params: Any = None) -> np.ndarray | dict | None:
        """
        Should return the visualization as np.array (Heatmap , RGB, Gray) or dict (Features <str, float>)
        :param m: the visualization method to query
        :param params: if ths method requires parameters, then this should be the dataclass of the parameters
        :return: visualization as np.array (Heatmap , RGB, Gray) or dict (Features <str, float>)
        """
        return None