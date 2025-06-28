import numbers
from typing import Optional, Callable
import numpy as np
from pearl.lab.annotations import Param


class test:
    integer: Param(int) = 45
    ranged_integer: Param(int, range_start=0, range_end=100) = 50
    floating: Param(float) = 1.4
    ranged_floating: Param(float, range_start=0.0, range_end=1.0) = 0.5
    string: Param(str) = "Hello world"
    string_file: Param(str, isFilePath=True, editable=False, default="C:/path/to/file.txt") = "C:/path/to/file.txt"
    mode: Param(int, choices=[0, 1, 2], default=1) = 1

    def __init__(self, x: int , y = None):
        pass


def func(x: int, y):
    """
    A simple function that does nothing.

    Args:
        x: An integer parameter.
        y: An optional string parameter with a default value.
    """
    pass

class A:
    pass

class B(A):
    pass

x = 45


from pearl.agent import RLAgent

class TorchDQN(RLAgent):
    """
    Simple DQN agent using PyTorch.
    """
    def __init__(self, model_path: str, module, device):
        print("hello from agent!")

    def predict(self, observation):
        pass

    def get_q_net(self):
        pass