from typing import Optional, Tuple, Callable, Dict, Any, Union
from pearl.env import RLEnvironment
import collections
import numpy as np
import gymnasium as gym
import ale_py

from pearl.lab.visual import VisualizationMethod
from pearl.lab.annotations import Param

gym.register_envs(ale_py)

class GymRLEnv(RLEnvironment):
    def __init__(
        self,
        env_name: str,
        stack_size: int = 4,
        frame_skip: int = 1,
        render_mode: Optional[str] = None,
        max_episode_steps: Optional[int] = None,
        reward_clipping: Optional[Tuple[float, float]] = None,
        reward_scaling: Optional[float] = None,
        observation_preprocessing: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        action_repeat: Optional[int] = None,
        seed: Optional[int] = None,
        record_video: bool = False,
        video_path: Optional[str] = None,
        normalize_observations: bool = False,
        num_envs: int = 1,
        tabular: bool = False
    ):
        super().__init__()
        # Create vectorized or single env
        if num_envs > 1:
            if tabular:
                self.env = gym.vector.make(
                    env_name,
                    num_envs=num_envs,
                    render_mode=render_mode
                )
            else:
                self.env = gym.vector.make(
                    env_name,
                    num_envs=num_envs,
                    frameskip=frame_skip,
                    render_mode=render_mode
                )
            self.observation_space = self.env.single_observation_space
            self.action_space = self.env.single_action_space
        else:
            if tabular:
                self.env = gym.make(
                    env_name,
                    render_mode=render_mode,
                )
            else: 
                self.env = gym.make(
                    env_name,
                    frameskip=frame_skip,
                    render_mode=render_mode
                )
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
        # Time limit wrapper
        if max_episode_steps:
            self.env = gym.wrappers.TimeLimit(self.env, max_episode_steps)
        # Video recording
        if record_video and video_path:
            self.env = gym.wrappers.RecordVideo(self.env, video_path)
        # Seed
        if seed is not None:
            self.env.reset(seed=seed)
        # Frame stack storage
        self.stack_size = stack_size
        self.frames = collections.deque(maxlen=stack_size)
        self.action_repeat = action_repeat or frame_skip
        self.reward_clipping = reward_clipping
        self.reward_scaling = reward_scaling
        self.observation_preprocessing = observation_preprocessing
        self.normalize_observations = normalize_observations

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.observation_preprocessing:
            frame = self.observation_preprocessing(frame)
        if self.normalize_observations:
            frame = frame.astype(np.float32) / 255.0
        return np.expand_dims(frame, -1)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options or {})
        raw = obs if not isinstance(obs, tuple) else obs[0]
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(self._process_frame(raw))
        if self.stack_size <= 1:
            return obs, info
        stacked = np.concatenate(self.frames, axis=-1)
        return stacked, info

    def step(
        self,
        action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], bool, bool, Dict[str, Any]]:
        total_reward = 0.0
        terminated = False
        truncated = False
        for _ in range(self.action_repeat):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward if isinstance(reward, float) or isinstance(reward, int) else reward[0]
            terminated = terminated or (term if isinstance(term, bool) else term[0])
            truncated = truncated or (trunc if isinstance(trunc, bool) else trunc[0])
        raw = obs if not isinstance(obs, tuple) else obs[0]
        processed = self._process_frame(raw)
        self.frames.append(processed)
        if self.stack_size > 1:
            stacked = np.concatenate(self.frames, axis=-1)
        else:
            stacked = processed
        if self.reward_clipping:
            total_reward = float(np.clip(total_reward, *self.reward_clipping))
        if self.reward_scaling:
            total_reward *= self.reward_scaling
        reward_out = {"reward": np.array(total_reward, np.float32)}
        return stacked, reward_out, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return self.env.render()

    def get_observations(self) -> np.ndarray:
        if self.stack_size <= 1:
            return np.squeeze(list(self.frames)[0], axis=-1)
        return np.concatenate(list(self.frames), axis=-1)

    def supports(self, m: VisualizationMethod) -> bool:
        if not isinstance(m, VisualizationMethod):
            m = VisualizationMethod(m)

        return m == VisualizationMethod.RGB_ARRAY


    def getVisualization(self, m: VisualizationMethod, params: Any = None) -> np.ndarray | dict | None:
        if not isinstance(m, VisualizationMethod):
            m = VisualizationMethod(m)
        if m == VisualizationMethod.RGB_ARRAY:
            rgb_image = self.render("rgb_array")
            if rgb_image.max() > 1.0:
                rgb_image = rgb_image.astype(np.float32) / 255.0
            return rgb_image
        return None

    def getVisualizationParamsType(self, m: VisualizationMethod) -> type | None:
        return None
