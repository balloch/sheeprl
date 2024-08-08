# from sheeprl.utils.imports import _IS_DMC_AVAILABLE

# if not _IS_DMC_AVAILABLE:
#     raise ModuleNotFoundError(_IS_DMC_AVAILABLE)

from typing import Any, Dict, Optional, SupportsFloat, Tuple, Union
import os

import gymnasium as gym
import numpy as np
import robosuite as suite
from gymnasium import spaces
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING  #*

## TODO It doesn't seem like this should be a wrapper, but just following DMC here
class RobosuiteWrapper(gym.Wrapper):
    def __init__(
        self,
        env_name: str,
        env_config: str,
        robot: str,
        bddl_file = None,
        controller: Any = 'OSC_POSE',
        hard_reset: bool = False,
        horizon: int = 500,
        reward_scale: float = 1.0,
        reward_shaping: bool = True,
        ignore_done: bool = True,
        has_renderer: bool = False,
        has_offscreen_renderer: bool = False,
        use_camera_obs: bool = False,
        control_freq: int = 20,
        keys=None,
        channels_first = True
        ):
        """Robosuite wrapper
        Args:
            env_name: (str),
            env_config: (str),
            robot: (str),
            controller: (Any) = 'OSC_POSE',
            hard_reset: (bool) = False,
            horizon: (int) = 500,
            reward_scale: (float) = 1.0,
            reward_shaping: (bool) = True,
            ignore_done: (bool) = True,
            has_renderer: (bool) = False,
            has_offscreen_renderer: (bool) = False,
            use_camera_obs: (bool) = False,
            control_freq: (int) = 20,
        """

        self.env_name = env_name
        self.env_config = env_config
        self.robot = robot
        self.bddl_file = bddl_file
        self.controller = controller
        self.hard_reset = hard_reset
        self.horizon = horizon
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.ignore_done = ignore_done
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.use_camera_obs = use_camera_obs
        self.control_freq = control_freq

        libero_args=dict(
            env_configuration=self.env_config,
            robots=[self.robot],
            controller_configs=suite.controllers.load_controller_config(default_controller=self.controller),
            hard_reset=self.hard_reset,
            horizon=self.horizon,
            reward_scale=self.reward_scale,
            reward_shaping=self.reward_shaping,
            ignore_done=self.ignore_done,
            has_renderer=self.has_renderer,
            has_offscreen_renderer=self.has_offscreen_renderer,
            use_camera_obs=self.use_camera_obs,
            control_freq=self.control_freq,
        )

        extra_robosuite_make_args=dict(
            env_name=self.env_name,
        )

# robosuite_make_args=dict(
#     env_name='PickPlace',
#     env_configuration='single-arm-opposed',
#     robots=['Panda'],
#     controller_configs=suite.controllers.load_controller_config(default_controller=controller),
#     hard_reset=False,
#     horizon=500,
#     reward_scale=1.0,
#     reward_shaping=True,
#     ignore_done=True,
#     has_renderer=False,
#     has_offscreen_renderer=False,
#     use_camera_obs=False,
#     control_freq=20,
# )
        # Create task environment
        if self.bddl_file:
            assert os.path.exists(bddl_file)
            problem_info = BDDLUtils.get_problem_info(bddl_file)
            env = TASK_MAPPING[problem_info["problem_name"]](
                bddl_file_name=bddl_file,
                **libero_args,
            )
        else:
            env = suite.make(**libero_args,
                             **extra_robosuite_make_args)

        super().__init__(env)

        obs = self.env.reset()
        obs_spec = self.env.observation_spec()

        self._height = obs['agentview_image'].shape[0]
        self._width = obs['agentview_image'].shape[1]
        self.name = self.robot + "_" + type(self.env).__name__
        self._from_pixels = self.env.use_camera_obs
        self._channels_first = channels_first

        ## Convert to Gym-style
        obs_space = {}
        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
                obs_shape = (3, self._height, self._width) if channels_first else (self._height, self._width, 3)
                obs_space["rgb"] = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
            for obs_key in obs.keys():
                if '_pos' in obs_key:
                    keys += [obs_key]
                    obs_space[obs_key] = spaces.Box(low=-1,
                                                high=1,
                                                shape=obs_spec[obs_key].shape,
                                                dtype=obs_spec[obs_key].dtype)
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
                if idx == 0:
                    obs_space["state"] = spaces.Box(low=-1,
                                                    high=1,
                                                    shape=obs_spec["robot{}_proprio-state".format(idx)].shape,
                                                    dtype=obs_spec["robot{}_proprio-state".format(idx)].dtype)  # TODO: does this need to be numpy?
                    # _spec_to_box(self.env.observation_spec().values(), np.float64)
                else:
                    obs_space[f"state{idx}"] = spaces.Box(low=-1,
                                                    high=1,
                                                    shape=obs_spec["robot{}_proprio-state".format(idx)].shape,
                                                    dtype=obs_spec["robot{}_proprio-state".format(idx)].dtype)  # TODO: does this need to be numpy?
        else:
            raise NotImplementedError
        self.keys = list(set(keys))

        self._from_vectors = 'robot0_proprio-state' in keys

        # # Get reward range
        # self.reward_range = (0, self.env.reward_scale)

        # true and normalized action spaces
        a_low, a_high = self.env.action_spec
        self._true_action_space = spaces.Box(low=a_low, high=a_high, dtype=np.float32)
        self._norm_action_space = spaces.Box(low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32)
        self._action_space = spaces.Box(low=a_low, high=a_high)

        # set the reward range
        self._reward_range = (0, self.env.reward_scale)  # self.reward_range

        # create observation space
        self._observation_space = spaces.Dict(obs_space)

        # state space
        self._state_space = obs_space["state"]  # Just copying the dmc implementation
        self.current_state = None
        # render
        self._render_mode: str = "rgb_array"
        # metadata
        self._metadata = {}
        # self._metadata = {"render_fps": 30}
        # set seed
        self.seed = 10


    def __getattr__(self, name):
        return getattr(self.env, name)

    def _get_obs(self, obs_data) -> Dict[str, np.ndarray]:
        obs = {}
        # at least one between from_pixels and from_vectors is True
        if self._from_pixels:
            rgb_obs = obs_data['agentview_image']
            if self._channels_first:
                rgb_obs = rgb_obs.transpose(2, 0, 1).copy()
            obs["rgb"] = rgb_obs
        if self._from_vectors:
            obs["state"] = obs_data["robot{}_proprio-state".format(0)]
        for obs_key in obs_data.keys():
            if '_pos' in obs_key:
                obs[obs_key] = obs_data[obs_key]
        return obs

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)


    # def _convert_action(self, action) -> np.ndarray:
    #     action = action.astype(np.float64)
    #     true_delta = self._true_action_space.high - self._true_action_space.low
    #     norm_delta = self._norm_action_space.high - self._norm_action_space.low
    #     action = (action - self._norm_action_space.low) / norm_delta
    #     action = action * true_delta + self._true_action_space.low
    #     action = action.astype(np.float32)
    #     return action

    @property
    def observation_space(self) -> Union[spaces.Dict, spaces.Box]:
        return self._observation_space

    @property
    def state_space(self) -> spaces.Box:
        return self._state_space

    @property
    def action_space(self) -> spaces.Box:
        return self._norm_action_space

    @property
    def reward_range(self) -> Tuple[float, float]:
        return self._reward_range

    @property
    def render_mode(self) -> str:
        return self._render_mode

    # def seed(self, seed: Optional[int] = None):
    #     self._true_action_space.seed(seed)
    #     self._norm_action_space.seed(seed)
    #     self._observation_space.seed(seed)

    def step(
        self, action: Any
    ) -> Tuple[Union[Dict[str, np.ndarray], np.ndarray], SupportsFloat, bool, bool, Dict[str, Any]]:
        # action = self._convert_action(action)
        time_step = self.env.step(action)  # observations, reward, done, info
        self.current_state = time_step[0]
        # self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step[0])
        reward = time_step[1]
        terminated = time_step[2]
        truncated = time_step[2]
        # terminated = truncated
        infos = time_step[3]
        infos["discount"] = .997  # TODO: I don't know if thats correct
        infos["internal_state"] = time_step[0]
        return obs, reward, terminated, truncated, infos

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Union[Dict[str, np.ndarray], np.ndarray], Dict[str, Any]]:
        if not isinstance(seed, np.random.RandomState):
            seed = np.random.RandomState(seed)
        # self.env.task._random = seed
        orig_obs = self.env.reset()
        self.current_state = orig_obs
        # self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(orig_obs)
        return obs, {}
        # return obs, (), False, False, {}

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()

    def render(self): # -> RenderFrame | list[RenderFrame] | None:
        # self.sim._render_context_offscreen
        return self.env._get_observations()['agentview_image']
