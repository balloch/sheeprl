import argparse
import os
import time
import datetime
import json
from glob import glob

import h5py
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import * #TASK_MAPPING


# from libero.scripts import *

def collect_trajectory(env, timesteps=1000, policy=None):
    """Run a random policy to collect trajectories.

    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment instance to collect trajectories from
        timesteps(int): how many environment timesteps to run for a given trajectory
    """

    env.reset()
    dof = env.action_dim
    if policy is None:
        print('Using random policy')
        rng = np.random.default_rng()
        action_range = env.action_spec[1] - env.action_spec[0]
        for t in range(timesteps):
            action = rng.random(size=env.action_dim)*action_range + env.action_spec[0]
            obs, rew, done, info = env.step(action)
            # env.render() # Necessary to save the images
            if timesteps>200 and t % 100 == 0:
                print(t)
            if done:
                break
    else:
        for t in range(timesteps):
            action = policy(obs)
            obs, rew, done, info = env.step(action)
            # env.render()
            if timesteps>200 and t % 100 == 0:
                print(t)
            if done:
                break

    # Save data with the reset call in DataCollectionWrapper
    env.reset()

    print("Done with trajectory collection")


def gather_demonstrations_as_hdf5(
    directory,
    out_dir,
    env_info,
    bddl_file,
    remove_directory=[],
    save_obs_keys=False,
):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        # print(ep_directory)
        if ep_directory in remove_directory:
            # print("Skipping")
            continue
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        if save_obs_keys:
            obs = {key: [] for key in save_obs_keys}

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])

        if len(states) == 0:
            continue

        # Delete the first actions and the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))


    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    grp.attrs["problem_info"] = json.dumps(problem_info)
    grp.attrs["bddl_file_name"] = bddl_file
    grp.attrs["bddl_file_content"] = str(open(bddl_file, "r", encoding="utf-8"))

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bddl-files",
        type=str,
        default='/home/balloch/code/LIBERO/libero/libero/bddl_files/libero_10'
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="/home/balloch/code/sheeprl/demonstration_data2",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--num-demonstration",
        type=int,
        default=10,
        help="number of demonstrations",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default=["Panda"],
        help="Which robot(s) to use in the env",
    )

    # parser.add_argument(
    #     "--save_obs_keys",
    #     nargs='+',
    #     type=str,
    #     default=[
    #         'agentview_image',
    #         'robot0_eye_in_hand_image',
    #         'robot0_eef_pos',

    #         ]
    # )
    parser.add_argument("--policy", type=str, default='random')
    parser.add_argument("--timesteps", type=int, default=128)

    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50734)

    args = parser.parse_args()
    # args.save_obs_keys = True

    if os.path.isdir(args.bddl_files):
        bddls = glob(args.bddl_files + '/*.bddl')
    else:
        bddls = [args.bddl_files]

    for bddl in bddls:
        assert os.path.exists(bddl)
        print(bddl)
        # Check if the directory already exists
        skip_directory = f"{args.policy}_{os.path.basename(bddl)[:-5]}"
        existing_dirs = [d for d in os.listdir(args.directory) if os.path.isdir(os.path.join(args.directory, d))]
        if any(skip_directory in d for d in existing_dirs):
            print(f"Directory {skip_directory} already exists as a substring in an existing directory. Skipping.")
            continue

        problem_info = BDDLUtils.get_problem_info(bddl)
        # Check if we're using a multi-armed environment and use env_configuration argument if so
        config = {}

        # Create environment
        problem_name = problem_info["problem_name"]
        domain_name = problem_info["domain_name"]
        if "TwoArm" in problem_name:
            config["env_configuration"] = args.config
        env = TASK_MAPPING[problem_name](
            bddl_file_name=bddl,
            has_renderer=False,
            robots=args.robots,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=True,
            reward_shaping=True,
            **config
        )

        # Grab reference to controller config and convert it to json-encoded string
        env_info = json.dumps(config)

        # wrap the environment with data collection wrapper
        tmp_directory = os.path.join(
            args.directory,
            'tmp',
            f'{problem_name}_{str(time.time()).replace(".", "_")}'
        )

        env = DataCollectionWrapper(
            env=env,
            directory=tmp_directory,
            # save_obs_keys=args.save_obs_keys,
        )

        # make a new timestamped directory
        t1, t2 = str(time.time()).split(".")
        new_dir = os.path.join(
            args.directory,
            f"{args.policy}_{os.path.basename(bddl)[:-5]}_{t1}_{t2}"
        )

        os.makedirs(new_dir)

        # collect demonstrations
        remove_directory = []
        i = 0

        while i < args.num_demonstration:
            print(i)
            collect_trajectory(
                env=env,
                timesteps=args.timesteps,
                policy=None if args.policy == 'random' else args.policy
            )
            i += 1
        # print(remove_directory)
        gather_demonstrations_as_hdf5(
            directory=tmp_directory,
            out_dir=new_dir,
            env_info=env_info,
            bddl_file=bddl,
            remove_directory=remove_directory,
            # save_obs_keys=args.save_obs_keys,
        )
