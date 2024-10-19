from typing import Any, Dict, Optional, SupportsFloat, Tuple, Union
from sheeprl.utils.utils import dotdict
import os

import gymnasium as gym
import numpy as np
import robosuite as suite
from gymnasium import spaces
import libero.libero.envs.bddl_utils as BDDLUtils
from robosuite.utils.sim_utils import check_contact
from libero.libero.envs import TASK_MAPPING  #*
import time
from functools import reduce
import operator



## TODO It doesn't seem like this should be a wrapper, but just following DMC here
class RobosuiteWrapper(gym.Wrapper):
    def __init__(
        self,
        env_name: str,
        env_config: str,
        robot: str,
        bddl_file: str = None,
        initial_joint_positions: list[float] | None = None,
        controller: Any = 'OSC_POSE',
        hard_reset: bool = False,
        horizon: int = 500,
        reward_scale: float = 1.0,
        reward_shaping: dotdict = None,
        ignore_done: bool = True,
        has_renderer: bool = False,
        has_offscreen_renderer: bool = False,
        use_camera_obs: bool = True,
        use_vector_obs: bool = False,
        control_freq: int = 20,
        keys: None = None,
        channels_first: bool = True,
        action_repeat: int = 1
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
        self.initial_joint_positions = initial_joint_positions
        self.controller = controller
        self.hard_reset = hard_reset
        self.horizon = horizon*action_repeat
        self.reward_scale = reward_scale
        self.reward_shaping = dotdict(reward_shaping)
        self.ignore_done = ignore_done
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.use_camera_obs = use_camera_obs
        self.use_vector_obs = use_vector_obs
        self.control_freq = control_freq
        self.action_repeat = action_repeat

        libero_args=dict(
            env_configuration=self.env_config,
            robots=[self.robot],
            controller_configs=suite.controllers.load_controller_config(default_controller=self.controller),
            hard_reset=self.hard_reset,
            horizon=self.horizon,
            reward_scale=self.reward_scale,
            reward_shaping=not self.reward_shaping.disable,
            ignore_done=self.ignore_done,
            has_renderer=self.has_renderer,
            has_offscreen_renderer=self.has_offscreen_renderer,
            use_camera_obs=self.use_camera_obs,
            control_freq=self.control_freq,
        )
        extra_robosuite_make_args=dict(
            env_name=self.env_name,
        )

        # Create task environment
        if self.bddl_file:
            assert os.path.exists(bddl_file)
            problem_info = BDDLUtils.get_problem_info(bddl_file)
            env = TASK_MAPPING[problem_info["problem_name"]](
                bddl_file_name=bddl_file,
                **libero_args,
            )
            self.concepts = np.array(get_bddl_concepts(self.bddl_file))
        else:
            env = suite.make(**libero_args,
                             **extra_robosuite_make_args)

        super().__init__(env)

        obs = self.env.reset()

        # We need this to be here because we want the environment to exist at this point
        if not self.reward_shaping.disable and self.bddl_file:
            # All our scenes have one goal, so this is simplified from the PickPlace implementation
            self._setup_staged_rewards(bddl_file)
            self._update_initial_distances()
            self._open_threshold = 0.02
            self._lift_threshold = 0.04

        if initial_joint_positions:
            self.env.robots[0].set_robot_joint_positions(self.initial_joint_positions)
            # Refresh observation without stepping
            obs = self.env._get_observations(force_update=True)

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
                obs_space["object-state"] = spaces.Box(low=-1,
                                                        high=1,
                                                        shape=obs_spec["object-state"].shape,
                                                        dtype=obs_spec["object-state"].dtype)
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
                obs_shape = (3, self._height, self._width) if channels_first else (self._height, self._width, 3)
                obs_space["agentview_rgb"] = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
            # Iterate over all robots to add to state
            if self.use_vector_obs:
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

        self._from_vectors = use_vector_obs  # 'robot0_proprio-state' in keys

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
        if "state" in obs_space:
            self._state_space = obs_space["state"]  # Just copying the dmc implementation
        self.current_state = None
        # render
        self._render_mode: str = "rgb_array"
        # metadata
        self._metadata = {}
        # self._metadata = {"render_fps": 30}
        # set seed
        self.seed = 10  #TODO this should come from the config

        self.step_returns = {
            'extrinsic': -1*np.ones(self.horizon*20),
            'intrinsic': {
                'reach': -1*np.ones(self.horizon*20),
                'grasp': -1*np.ones(self.horizon*20),
                'lift': -1*np.ones(self.horizon*20),
                'hover': -1*np.ones(self.horizon*20)
            }
        }
        self.ep_stats = {
            'length': [],
            'extrinsic': [],
            'intrinsic': {
                'reach': {
                    'mean': [],
                    'max': [],
                    },
                'grasp': {
                    'mean': [],
                    'max': [],
                    },
                'lift': {
                    'mean': [],
                    'max': [],
                    },
                'hover': {
                    'mean': [],
                    'max': [],
                    },
            }
        }
        self.ep_length = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _get_obs(self, obs_data) -> Dict[str, np.ndarray]:
        obs = {}
        # at least one between from_pixels and from_vectors is True
        if self._from_pixels:
            rgb_obs = obs_data['agentview_image']
            if self._channels_first:
                rgb_obs = rgb_obs.transpose(2, 0, 1).copy()
            obs["agentview_rgb"] = rgb_obs
        if self._from_vectors:
            obs["state"] = obs_data["robot{}_proprio-state".format(0)]
        if self.env.use_object_obs:
            obs["object-state"] = obs_data["object-state"]
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
        # Final reward scaling on truncation
        if time_step[2]:
            reward = reward * self.ep_length
        # try:
        self.step_returns['extrinsic'][self.ep_length] = reward
        
        reward = self.compute_reward()
        
        # except Exception as e:
        #     import pdb; pdb.set_trace()
        # if self.reward_shaping and self.bddl_file:
        #     reward += self.staged_rewards()
        terminated = time_step[2]
        truncated = time_step[2]
        # terminated = truncated
        infos = time_step[3]
        infos["discount"] = .997  # TODO: I don't know if thats correct
        infos["internal_state"] = time_step[0]
        
        
        infos["concepts"] = self.concepts

        self.ep_length += 1

        return obs, reward, terminated, truncated, infos

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Union[Dict[str, np.ndarray], np.ndarray], Dict[str, Any]]:
        if not isinstance(seed, np.random.RandomState):
            seed = np.random.RandomState(seed)
        # self.env.task._random = seed
        self.ep_start_times = time.perf_counter()
        # self.ep_returns = []  # np.zeros(self.num_envs, dtype=np.float32)

        # Accumulate stats
        if (self.step_returns['extrinsic'] > -1).any():
            self.ep_stats['extrinsic'].append(
                self.step_returns['extrinsic'][self.step_returns['extrinsic'] > -1].mean())
            for key in self.ep_stats['intrinsic'].keys():
                self.ep_stats['intrinsic'][key]['mean'].append(
                    self.step_returns['intrinsic'][key][self.step_returns['intrinsic'][key] > -1].mean())
                self.ep_stats['intrinsic'][key]['max'].append(
                    self.step_returns['intrinsic'][key][self.step_returns['intrinsic'][key] > -1].max())

            self.ep_stats['length'].append(self.ep_length)

            # Reset step statistics
            self.step_returns = {
                'extrinsic': -1*np.ones(self.horizon),
                'intrinsic': {
                    'reach': -1*np.ones(self.horizon),
                    'grasp': -1*np.ones(self.horizon),
                    'lift': -1*np.ones(self.horizon),
                    'hover': -1*np.ones(self.horizon)
                }
            }
            self.ep_length = 0

        orig_obs = self.env.reset()
        if self.initial_joint_positions:
            self.env.robots[0].set_robot_joint_positions(self.initial_joint_positions)
            # Resample without stepping
            orig_obs = self.env._get_observations(force_update=True)

        if not self.reward_shaping.disable and self.bddl_file:
            self._update_initial_distances()

        # Reset staged rewards accumulator
        if not self.reward_shaping.disable and self.bddl_file:
            self.max_lift_ep = 0

        self.current_state = orig_obs
        # self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(orig_obs)
        return obs, {}
        # return obs, (), False, False, {}

    def get_env_stats(self, stats_dict = None, reset: bool = False) -> Dict[str, Any]:
        if len(self.ep_stats['length']) == 0:
            return None
        elif stats_dict is not None:
            def get_nested_value(d, keys):
                try:
                    return reduce(operator.getitem, keys, d)
                except KeyError:
                    return None  # or handle the error as needed

            tracked_stats = {key: get_nested_value(self.ep_stats, value) for key, value in stats_dict.items()}
        else:
            tracked_stats = self.ep_stats.copy()

        if reset:
            self.ep_stats = {
                'length': [],
                'extrinsic': [],
                'intrinsic': {
                    'reach': {
                        'mean': [],
                        'max': [],
                        },
                    'grasp': {
                        'mean': [],
                        'max': [],
                        },
                    'lift': {
                        'mean': [],
                        'max': [],
                        },
                    'hover': {
                        'mean': [],
                        'max': [],
                        },
                }
            }
        return tracked_stats

    def _update_initial_distances(self):
        env : gym.Env = self.env
        # Calculate starting distances to normalize
        target_to_eef = env._gripper_to_target(
            gripper=env.robots[0].gripper,
            target=self._target_object['object'].root_body,
            target_type="body",
            return_distance=True
        )

        goal_xyz = env.sim.data.body_xpos[self._goal_location['body_geom_id']]
        object_xyz = env.sim.data.body_xpos[self._target_object['body_geom_id']]
        target_to_goal_xy = np.linalg.norm(goal_xyz[:2] - object_xyz[:2])
        target_to_goal_xyz = np.linalg.norm(goal_xyz - object_xyz)

        object_z = env.sim.data.body_xpos[self._target_object['body_geom_id']][2]

        self._initial_distances = {
            'target_to_eef': target_to_eef,
            'target_to_goal_xy': target_to_goal_xy,
            'target_to_goal_xyz': target_to_goal_xyz,   
            'object_z': object_z
        }

    def _setup_staged_rewards(self, bddl_file : str):
        env : gym.Env = self.env

        goal_state = BDDLUtils.robosuite_parse_problem(bddl_file)['goal_state'][0]
                # Hover reward relies only on xy position so in/on is identical
        target_object = goal_state[1]
        goal_object = goal_state[2].replace('_contain_region', '')

        # Saved important object information
        target_object = env.objects_dict[target_object]
        goal_object = env.objects_dict[goal_object]
        self._target_object = {
            'object': target_object,
            'body_geom_id': env.sim.model.body_name2id(target_object.root_body)
        }
        self._goal_location = {
            'object': goal_object,
            'body_geom_id': env.sim.model.body_name2id(goal_object.root_body)
        }

        self.max_lift_ep = 0

    def staged_rewards(self):
        """
        Function to calculate staged rewards
        Adapted from the function of the same name in robosuite.environments.manipulation.pick_place
        """
        assert self.reward_shaping.disable == False
        # The suite.make environment implements its own staged rewards
        assert self.bddl_file

        reach_mult = self.reward_shaping.stages.reach.mult
        # Grasp multiplier set to zero (reference sets it to 0.35) but implemented for completeness
        grasp_mult = self.reward_shaping.stages.grasp.mult
        hover_mult = self.reward_shaping.stages.hover.mult
        lift_mult = self.reward_shaping.stages.lift.mult

        r_reach = 0
        # Normalized, > 1 if farther than initial position, [0, 1] if closer
        eef_to_target_dist = self.env._gripper_to_target(
            gripper=self.env.robots[0].gripper,
            target=self._target_object['object'].root_body,
            target_type="body",
            return_distance=True
        ) / self._initial_distances['target_to_eef']

        # Grasping reward
        # Grows as independent approaches zero
        r_reach = ((1 - eef_to_target_dist) ** 2) * reach_mult
        # Constrain to [0, reach_mult]
        # If statement to prevent reward from increasing if dist > 1
        r_reach = max(0, min(r_reach, reach_mult)) if eef_to_target_dist <= 1 else 0

        is_left_contact = check_contact(self.env.sim, names[0], self._target_object['object'])
        is_right_contact = check_contact(self.env.sim, names[1], self._target_object['object'])
        is_touching = is_left_contact and is_right_contact
        
        finger1_col = self.env.sim.data.geom_xpos[self.env.sim.model.geom_name2id("gripper0_finger1_collision")]
        finger2_col = self.env.sim.data.geom_xpos[self.env.sim.model.geom_name2id("gripper0_finger2_collision")]
        is_open = np.linalg.norm(finger1_col - finger2_col) > 0.02 # Magic number, model starts at 0.06419 (significantly open)    
        
        # As per the isaac cube stack definition
        is_grasping = is_open and is_touching

        # Normalized
        goal_xy = self.env.sim.data.body_xpos[self._goal_location['body_geom_id']][:2]
        object_xy = self.env.sim.data.body_xpos[self._target_object['body_geom_id']][:2]
        target_to_goal_dist = np.linalg.norm(goal_xy - object_xy) / self._initial_distances['target_to_goal_xy']

        # Grasping reward
        r_grasp = 0.0
        if is_grasping:
            # x : distance to goal
            # grasping reward decreases as one gets closer to the goal
            r_grasp += (- (1 - target_to_goal_dist) ** 2 + 1) * grasp_mult
        # Constrain to [0, 0.5 * grasp_mult]
        # If statement prevents reward from decreasing if target_to_goal > 1
        r_grasp = max(0, min(r_grasp, grasp_mult)) if target_to_goal_dist <= 1 else grasp_mult

        r_lift = 0.0

        # Lift reward
        distance_lifted = self.env.sim.data.body_xpos[self._target_object['body_geom_id']][2] - self._initial_distances['object_z']
        distance_lifted = max(distance_lifted, 0) # Safeguard for dropping the object below initial position
        # l(z) = tanh(4z) * pi/32
        # z : distance lifted
        r_lift = np.tanh(4 * distance_lifted) * lift_mult
        # Penalty is applied as object gets closer to goal
        # p(x) = tanh(20x)
        # Goes to zero as x -> 0
        lift_interpolate_progress = np.tanh(10 * (target_to_goal_dist + 1 / 50))
        r_lift = r_lift * lift_interpolate_progress + (1 - lift_interpolate_progress) * 0.08

        # Constrain to [0, pi/32]
        r_lift = max(0, min(r_lift, lift_mult))

        self.max_lift_ep = max(self.max_lift_ep, r_lift)

        r_hover = 0.0
        if self.max_lift_ep > 0.03:
            # Hover reward
            r_hover = ((1.75 - 1.75  * target_to_goal_dist)**2) * hover_mult

            # If prevents reward from increasing if target_to_goal > 1
            # Max is derived from h(0)
            r_hover = max(0, min(r_hover, 3.0625 * hover_mult)) if target_to_goal_dist <= 1 else 0

        # Uncomment if you need this at some point
        # print("distance_to_goal: {}".format(target_to_goal_dist))
        # print("r_reach: {}, r_grasp: {}, r_lift: {}, r_hover: {}".format(r_reach, r_grasp, r_lift, r_hover))

        return r_reach, r_grasp, r_lift, r_hover

    def nvidia_staged_rewards(self):
        """
        Computes isaac-style dense rewards
        Uses robosuite functions to calculate distance
        """
        
        reward = 0.0
    
        # Reaching reward
        # Normalized, > 1 if farther than initial position, [0, 1] if closer
        eef_to_target_dist = self.env._gripper_to_target(
            gripper=self.env.robots[0].gripper,
            target=self._target_object['object'].root_body,
            target_type="body",
            return_distance=True
        ) / self._initial_distances['target_to_eef']
        
        reward = reach_reward = 2 * (1 - np.tanh(5 * eef_to_target_dist))
        
        # Grasp and place reward
        goal_xy = self.env.sim.data.body_xpos[self._goal_location['body_geom_id']][:2]
        object_xy = self.env.sim.data.body_xpos[self._target_object['body_geom_id']][:2]
        target_to_goal_dist = np.linalg.norm(goal_xy - object_xy) / self._initial_distances['target_to_goal_xy']
        
        # [0, 1]
        place_reward = 1 - np.tanh(5.0 * target_to_goal_dist)
        
        names = ['gripper0_finger1_pad_collision', 'gripper0_finger2_pad_collision']
        
        # Grasp
        # Requires contact with BOTH pads and hand at least a bit open
        is_left_contact = check_contact(self.env.sim, names[0], self._target_object['object'])
        is_right_contact = check_contact(self.env.sim, names[1], self._target_object['object'])
        is_touching = is_left_contact and is_right_contact
        
        finger1_col = self.env.sim.data.geom_xpos[self.env.sim.model.geom_name2id("gripper0_finger1_collision")]
        finger2_col = self.env.sim.data.geom_xpos[self.env.sim.model.geom_name2id("gripper0_finger2_collision")]
        is_open = np.linalg.norm(finger1_col - finger2_col) > self._open_threshold # Magic number, model starts at 0.06419 (significantly open)    
        
        # Lift reward
        distance_lifted = self.env.sim.data.body_xpos[self._target_object['body_geom_id']][2] - self._initial_distances['object_z']
        distance_lifted = max(distance_lifted, 0) # Safeguard for dropping the object below initial position
        # z : distance lifted
        r_lift = distance_lifted > self._lift_threshold
        
        # As per the isaac cube stack definition
        if is_touching and is_open:
            reward = 4
            if r_lift:
                reward += r_lift + place_reward
            
        return reward, reach_reward, 4 if is_touching and is_open else 0, r_lift, place_reward 

    # kwargs here to keep compatibility with gym inteface
    def compute_reward(self, achieved_goal = None, desired_goal = None, info = None):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        
        reward = self.env.reward()
        
        if not self.reward_shaping.disable and self.bddl_file:
            if self.reward_shaping.mode == 'summed': # TODO: Change this name (mode : str = 'summed' | 'stepped' | 'nvidia')
                r_reach, r_grasp, r_lift, r_hover = self.staged_rewards()
                reward += sum([r_reach, r_grasp, r_lift, r_hover])
            elif self.reward_shaping.mode == 'stepped':
                r_reach, r_grasp, r_lift, r_hover = self.staged_rewards()
                tol = self.reward_shaping.tolerance
                reach_mult, grasp_mult, lift_mult, hover_mult = (
                    self.reward_shaping.stages.reach.mult,
                    self.reward_shaping.stages.grasp.mult,
                    self.reward_shaping.stages.lift.mult,
                    self.reward_shaping.stages.hover.mult,
                )
                grasp_step, lift_step, hover_step = (
                    self.reward_shaping.stages.grasp.step,
                    self.reward_shaping.stages.lift.step,
                    self.reward_shaping.stages.hover.step
                )
                if r_hover >= tol:
                    reward += r_hover + sum((reach_mult, grasp_mult, lift_mult)) + sum((grasp_step, lift_step, hover_step)) # Saturating previous rewards
                    # print("hover")
                elif r_lift >= tol:
                    reward += r_lift + sum((reach_mult, grasp_mult)) + sum((grasp_step, lift_step))
                    # print("lift")
                elif r_grasp >= tol:
                    reward += r_grasp + reach_mult + grasp_step
                    # print("grasp")
                else:
                    reward += r_reach
                    # print("reach")   
            elif self.reward_shaping.mode == 'nvidia':
                dense_reward, r_reach, r_grasp, r_lift, r_hover = self.nvidia_staged_rewards()
                reward += dense_reward
                
            staged_rewards = {
                'reach': r_reach,
                'grasp': r_grasp,
                'lift': r_lift,
                'hover': r_hover
            }
            # print([(k, v) for k, v in staged_rewards.items()])
            # print(reward)
            for key in staged_rewards.keys():
                # try:
                self.step_returns['intrinsic'][key][self.ep_length] = staged_rewards[key]
                # except Exception as e:
                #     import pdb; pdb.set_trace()

        return reward

    def render(self): # -> RenderFrame | list[RenderFrame] | None:
        # self.sim._render_context_offscreen
        return self.env._get_observations()['agentview_image']


concept_dict = {
    'white_yellow_mug': 0,
    'butter': 1,
    'wine_bottle': 2,
    'yellow_book': 3,
    'ketchup': 4,
    'tomato_sauce': 5,
    'orange_juice': 6,
    'porcelain_mug': 7,
    'chefmate_8_frypan': 8,
    'cream_cheese': 9,
    'plate': 10,
    'chocolate_pudding': 11,
    'red_coffee_mug': 12,
    'moka_pot': 13,
    'basket': 14,
    'milk': 15,
    'white_bowl': 16,
    'wooden_tray': 17,
    'akita_black_bowl': 18,
    'alphabet_soup': 19,
    'black_book': 20,
    'new_salad_dressing': 21,
    'bbq_sauce': 22,  #Not in libero90
    # 'cookies': 23,  #Not in libero90
    # 'glazed_rim_porcelain_ramekin': 24,  # Not in libero90
}


def get_bddl_concepts(file_name):
    concept_list = []
    with open(file_name, 'r') as bddl_file:
        all_lines = bddl_file.readlines()
    extract = False
    for line in all_lines:
        line = line.strip()
        if line == ")":
            extract = False
        if extract:
            line = line.split('-')[-1].strip()
            concept_list.append(line)
        if line == "(:objects":
            extract = True

    arr = np.zeros(len(concept_dict.keys()))
    concept_list = [concept_dict[c] for c in concept_list]
    # with open('concepts.txt', 'a') as f:
    #   for line in concept_list:
    #       f.write(f"{line}\n")
    arr[concept_list] = 1
    return arr.tolist()
