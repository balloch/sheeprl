import os
import time
from gym import spaces
import robosuite as suite

from libero.libero.envs import * #TASK_MAPPING
import libero.libero.envs.bddl_utils as BDDLUtils


os.environ['MUJOCO_GL'] = 'egl'
if __name__ == '__main__':

    bddl='/home/balloch/code/LIBERO/libero/libero/bddl_files/libero_90/STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf.bddl'

    problem_info = BDDLUtils.get_problem_info(bddl)
    # Check if we're using a multi-armed environment and use env_configuration argument if so
    config = {"policy": "random"}

    # Create environment
    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]


    env = TASK_MAPPING[problem_name](
        bddl_file_name=bddl,
        has_renderer=False,
        robots=["Panda"],
        has_offscreen_renderer=True,
        # render_camera="agentview",
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        render_gpu_device_id=0,
    )


    # env = suite.make(
    #     env_name='Wipe',
    #     robots="Panda",
    #     has_renderer=False,
    #     has_offscreen_renderer=True,
    #     use_camera_obs=True,
    #     camera_heights=84,
    #     camera_widths=84,
    #     camera_names='sideview',
    #     render_gpu_device_id=0,
    #     )


    obs, done = env.reset(), False
    low, high = env.action_spec
    action_space = spaces.Box(low=low, high=high)
    steps, time_stamp = 0, time.time()
    while True:
        while not done and steps < 1000:
            obs, reward, done, info = env.step(action_space.sample())
            steps += 1
        obs, done = env.reset(), False
        print(f'FPS: {steps / (time.time() - time_stamp)}')
        steps, time_stamp = 0, time.time()
