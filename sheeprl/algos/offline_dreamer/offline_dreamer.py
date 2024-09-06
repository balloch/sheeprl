"""Dreamer-V3 implementation from [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)
Adapted from the original implementation from https://github.com/danijar/dreamerv3
"""

from __future__ import annotations

import copy
import os
import warnings
from functools import partial
from typing import Any, Callable, Dict, Sequence, Tuple, Union
import pathlib

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor
from torch.distributions import Distribution, Independent, OneHotCategorical
from torch.optim import Optimizer
from torchmetrics import SumMetric

from sheeprl.algos.offline_dreamer.agent import WorldModel, CBWM, build_agent
from sheeprl.algos.offline_dreamer.loss import reconstruction_loss
from sheeprl.algos.offline_dreamer.utils import Moments, compute_lambda_values, prepare_obs, test
from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer
from sheeprl.envs.wrappers import RestartOnException
from sheeprl.envs.robosuite import get_bddl_concepts
from sheeprl.utils.distribution import (
    BernoulliSafeMode,
    MSEDistribution,
    SymlogDistribution,
    TwoHotEncodingDistribution,
)
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import Ratio, save_configs

# Decomment the following two lines if you cannot start an experiment with DMC environments
# os.environ["PYOPENGL_PLATFORM"] = ""
# os.environ["MUJOCO_GL"] = "osmesa"


def dynamic_learning(
    world_model: Union[WorldModel, CBWM],
    data: Dict[str, Tensor],
    batch_actions: Tensor,
    embedded_obs: Dict[str, Tensor],
    stochastic_size: int,
    discrete_size: int,
    recurrent_state_size: int,
    batch_size: int,
    sequence_length: int,
    decoupled_rssm: bool,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    # Dynamic Learning
    stoch_state_size = stochastic_size * discrete_size
    recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)
    recurrent_states = torch.empty(sequence_length, batch_size, recurrent_state_size, device=device)
    priors_logits = torch.empty(sequence_length, batch_size, stoch_state_size, device=device)

    if decoupled_rssm:
        posteriors_logits, posteriors = world_model.rssm._representation(embedded_obs)
        for i in range(0, sequence_length):
            if i == 0:
                posterior = torch.zeros_like(posteriors[:1])
            else:
                posterior = posteriors[i - 1 : i]
            recurrent_state, posterior_logits, prior_logits = world_model.rssm.dynamic(
                posterior,
                recurrent_state,
                batch_actions[i : i + 1],
                data["is_first"][i : i + 1],
            )
            recurrent_states[i] = recurrent_state
            priors_logits[i] = prior_logits
    else:
        posterior = torch.zeros(1, batch_size, stochastic_size, discrete_size, device=device)
        posteriors = torch.empty(sequence_length, batch_size, stochastic_size, discrete_size, device=device)
        posteriors_logits = torch.empty(sequence_length, batch_size, stoch_state_size, device=device)
        for i in range(0, sequence_length):
            recurrent_state, posterior, _, posterior_logits, prior_logits = world_model.rssm.dynamic(
                posterior,
                recurrent_state,
                batch_actions[i : i + 1],
                embedded_obs[i : i + 1],
                data["is_first"][i : i + 1],
            )
            recurrent_states[i] = recurrent_state
            priors_logits[i] = prior_logits
            posteriors[i] = posterior
            posteriors_logits[i] = posterior_logits

    latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1)
    cem_data = None
    if isinstance(world_model, CBWM):
        # print("DYNAMIC LEARNING!!!!!!!!!")
        random_latent = world_model.cem.sample_latent(list(latent_states.size()))
        latent_states, pred_concepts, real_concept_latent, real_non_concept_latent = world_model.cem(latent_states)
        _, _, rand_concept_latent, rand_non_concept_latent = world_model.cem(random_latent)
        if data.get("targets") is not None:
            target_concepts = data["targets"]
        else:
            target_concepts = None
        cem_data = [pred_concepts,
                    target_concepts,
                    real_concept_latent,  # This is the stuff we are comparing
                    real_non_concept_latent,
                    rand_concept_latent,
                    rand_non_concept_latent]
    return latent_states, priors_logits, posteriors_logits, posteriors, recurrent_states, cem_data


def behaviour_learning(
    posteriors: torch.Tensor,
    recurrent_states: torch.Tensor,
    data: Dict[str, torch.Tensor],
    world_model: Union[WorldModel, CBWM],
    actor: _FabricModule,
    stoch_state_size: int,
    recurrent_state_size: int,
    batch_size: int,
    sequence_length: int,
    horizon: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    imagined_prior = posteriors.detach().reshape(1, -1, stoch_state_size)
    recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
    imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
    if isinstance(world_model, CBWM):
        # print("BEHAVIOR LEARNING!!!!!!!!!")
        imagined_latent_state, _, _, _ = world_model.cem(imagined_latent_state)

    imagined_trajectories = torch.empty(
        horizon + 1,
        batch_size * sequence_length,
        imagined_latent_state.size()[-1],
        device=device,
    )
    imagined_trajectories[0] = imagined_latent_state
    imagined_actions = torch.empty(
        horizon + 1,
        batch_size * sequence_length,
        data["actions"].shape[-1],
        device=device,
    )
    actions_list, _ = actor(imagined_latent_state.detach())
    actions = torch.cat(actions_list, dim=-1)
    imagined_actions[0] = actions

    # The imagination goes like this, with H=3:
    # Actions:           a'0      a'1      a'2     a'4
    #                    ^ \      ^ \      ^ \     ^
    #                   /   \    /   \    /   \   /
    #                  /     \  /     \  /     \ /
    # States:        z0 ---> z'1 ---> z'2 ---> z'3
    # Rewards:       r'0     r'1      r'2      r'3
    # Values:        v'0     v'1      v'2      v'3
    # Lambda-values:         l'1      l'2      l'3
    # Continues:     c0      c'1      c'2      c'3
    # where z0 comes from the posterior, while z'i is the imagined states (prior)

    # Imagine trajectories in the latent space
    for i in range(1, horizon + 1):
        imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
        imagined_prior = imagined_prior.view(1, -1, stoch_state_size)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        if isinstance(world_model, CBWM):
            imagined_latent_state, _, _, _ = world_model.cem(imagined_latent_state)

        imagined_trajectories[i] = imagined_latent_state
        actions_list, _ = actor(imagined_latent_state.detach())
        actions = torch.cat(actions_list, dim=-1)
        imagined_actions[i] = actions

    return imagined_trajectories, imagined_actions


def train(
    fabric: Fabric,
    world_model: Union[WorldModel, CBWM],
    actor: _FabricModule,
    critic: _FabricModule,
    target_critic: torch.nn.Module,
    world_optimizer: Optimizer,
    actor_optimizer: Optimizer,
    critic_optimizer: Optimizer,
    data: Dict[str, Tensor],
    aggregator: MetricAggregator | None,
    cfg: Dict[str, Any],
    is_continuous: bool,
    actions_dim: Sequence[int],
    moments: Moments,
    compiled_dynamic_learning: Callable,
    compiled_behaviour_learning: Callable | None,
    compiled_compute_lambda_values: Callable,
) -> None:
    """Runs one-step update of the agent.

    Args:
        fabric (Fabric): the fabric instance.
        world_model (_FabricModule): the world model wrapped with Fabric.
        actor (_FabricModule): the actor model wrapped with Fabric.
        critic (_FabricModule): the critic model wrapped with Fabric.
        target_critic (nn.Module): the target critic model.
        world_optimizer (Optimizer): the world optimizer.
        actor_optimizer (Optimizer): the actor optimizer.
        critic_optimizer (Optimizer): the critic optimizer.
        data (Dict[str, Tensor]): the batch of data to use for training.
        aggregator (MetricAggregator, optional): the aggregator to print the metrics.
        cfg (DictConfig): the configs.
        is_continuous (bool): whether or not the environment is continuous.
        actions_dim (Sequence[int]): the actions dimension.
        moments (Moments): the moments for normalizing the lambda values.
    """
    # The environment interaction goes like this:
    # Actions:           a0       a1       a2      a4
    #                    ^ \      ^ \      ^ \     ^
    #                   /   \    /   \    /   \   /
    #                  /     v  /     v  /     v /
    # Observations:  o0       o1       o2       o3
    # Rewards:       0        r1       r2       r3
    # Dones:         0        d1       d2       d3
    # Is-first       1        i1       i2       i3

    batch_size = cfg.algo.per_rank_batch_size
    sequence_length = cfg.algo.per_rank_sequence_length
    recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size
    stochastic_size = cfg.algo.world_model.stochastic_size
    discrete_size = cfg.algo.world_model.discrete_size
    device = fabric.device
    batch_obs = {k: data[k] / 255.0 - 0.5 for k in cfg.algo.cnn_keys.encoder}
    batch_obs.update({k: data[k] for k in cfg.algo.mlp_keys.encoder})
    data["is_first"][0, :] = torch.ones_like(data["is_first"][0, :])

    # Given how the environment interaction works, we remove the last actions
    # and add the first one as the zero action
    batch_actions = torch.cat((torch.zeros_like(data["actions"][:1]), data["actions"][:-1]), dim=0)

    # Dynamic Learning
    stoch_state_size = stochastic_size * discrete_size

    # Embed observations from the environment
    embedded_obs = world_model.encoder(batch_obs)

    # Dynamic Learning
    ## TODO error here:
    # call_function <built-in function mul>(*(FakeTensor(..., device='cuda:0', size=(1, 16, 1)), FakeTensor(..., device='cuda:0', size=(1, s0, 7))), **{}):
    # The size of tensor a (16) must match the size of tensor b (s0) at non-singleton dimension 1)
    latent_states, priors_logits, posteriors_logits, posteriors, recurrent_states, cem_data = compiled_dynamic_learning(
        world_model,
        data,
        batch_actions,
        embedded_obs,
        stochastic_size,
        discrete_size,
        recurrent_state_size,
        batch_size,
        sequence_length,
        cfg.algo.world_model.decoupled_rssm,
        device,
    )

    # Compute predictions for the observations
    reconstructed_obs: Dict[str, torch.Tensor] = world_model.observation_model(latent_states)

    # Compute the distribution over the reconstructed observations
    po = {
        k: MSEDistribution(reconstructed_obs[k], dims=len(reconstructed_obs[k].shape[2:]))
        for k in cfg.algo.cnn_keys.decoder
    }
    po.update(
        {
            k: SymlogDistribution(reconstructed_obs[k], dims=len(reconstructed_obs[k].shape[2:]))
            for k in cfg.algo.mlp_keys.decoder
        }
    )

    # Compute the distribution over the rewards
    pr = TwoHotEncodingDistribution(world_model.reward_model(latent_states), dims=1)

    # Compute the distribution over the terminal steps, if required
    pc = Independent(BernoulliSafeMode(logits=world_model.continue_model(latent_states)), 1)
    continues_targets = 1 - data["terminated"]

    # Reshape posterior and prior logits to shape [B, T, 32, 32]
    priors_logits = priors_logits.view(*priors_logits.shape[:-1], stochastic_size, discrete_size)
    posteriors_logits = posteriors_logits.view(*posteriors_logits.shape[:-1], stochastic_size, discrete_size)

    # World model optimization step. Eq. 4 in the paper
    world_optimizer.zero_grad(set_to_none=True)
    rec_loss, loss_dict = reconstruction_loss(
        po,
        batch_obs,
        pr,
        data["rewards"],
        priors_logits,
        posteriors_logits,
        world_model,
        cem_data,
        cfg.algo.world_model.cbm_model.use_cbm,
        cfg.algo.world_model.kl_dynamic,
        cfg.algo.world_model.kl_representation,
        cfg.algo.world_model.kl_free_nats,
        cfg.algo.world_model.kl_regularizer,
        pc,
        continues_targets,
        cfg.algo.world_model.continue_scale_factor,
        cfg.algo.world_model.cbm_model.ortho_reg,
        cfg.algo.world_model.cbm_model.concept_reg
    )
    kl = loss_dict['kl']
    state_loss = loss_dict['kl_loss']
    reward_loss = loss_dict['reward_loss']
    observation_loss = loss_dict['observation_loss']
    continue_loss = loss_dict['continue_loss']
    fabric.backward(rec_loss)
    world_model_grads = None
    if cfg.algo.world_model.clip_gradients is not None and cfg.algo.world_model.clip_gradients > 0:
        world_model_grads = fabric.clip_gradients(
            module=world_model,
            optimizer=world_optimizer,
            max_norm=cfg.algo.world_model.clip_gradients,
            error_if_nonfinite=False,
        )
    world_optimizer.step()

    # Behaviour Learning
    if compiled_behaviour_learning is not None:
        imagined_trajectories, imagined_actions = compiled_behaviour_learning(
            posteriors,
            recurrent_states,
            data,
            world_model,
            actor,
            stoch_state_size,
            recurrent_state_size,
            batch_size,
            sequence_length,
            cfg.algo.horizon,
            device,
        )

        # Predict values, rewards and continues
        predicted_values = TwoHotEncodingDistribution(critic(imagined_trajectories), dims=1).mean
        predicted_rewards = TwoHotEncodingDistribution(world_model.reward_model(imagined_trajectories), dims=1).mean
        continues = Independent(BernoulliSafeMode(logits=world_model.continue_model(imagined_trajectories)), 1).mode
        true_continue = (1 - data["terminated"]).flatten().reshape(1, -1, 1)
        continues = torch.cat((true_continue, continues[1:]))

        # Estimate lambda-values
        lambda_values = compiled_compute_lambda_values(
            predicted_rewards[1:],
            predicted_values[1:],
            continues[1:] * cfg.algo.gamma,
            lmbda=cfg.algo.lmbda,
        )

        # Compute the discounts to multiply the lambda values to
        with torch.no_grad():
            discount = torch.cumprod(continues * cfg.algo.gamma, dim=0) / cfg.algo.gamma

        # Actor optimization step. Eq. 11 from the paper
        # Given the following diagram, with H=3
        # Actions:          [a'0]    [a'1]    [a'2]    a'3
        #                    ^ \      ^ \      ^ \     ^
        #                   /   \    /   \    /   \   /
        #                  /     \  /     \  /     \ /
        # States:       [z0] -> [z'1] -> [z'2] ->  z'3
        # Values:       [v'0]   [v'1]    [v'2]     v'3
        # Lambda-values:        [l'1]    [l'2]    [l'3]
        # Entropies:    [e'0]   [e'1]    [e'2]
        actor_optimizer.zero_grad(set_to_none=True)
        policies: Sequence[Distribution] = actor(imagined_trajectories.detach())[1]

        baseline = predicted_values[:-1]
        offset, invscale = moments(lambda_values, fabric)
        normed_lambda_values = (lambda_values - offset) / invscale
        normed_baseline = (baseline - offset) / invscale
        advantage = normed_lambda_values - normed_baseline
        if is_continuous:
            objective = advantage
        else:
            objective = (
                torch.stack(
                    [
                        p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]
                        for p, imgnd_act in zip(policies, torch.split(imagined_actions, actions_dim, dim=-1))
                    ],
                    dim=-1,
                ).sum(dim=-1)
                * advantage.detach()
            )
        try:
            entropy = cfg.algo.actor.ent_coef * torch.stack([p.entropy() for p in policies], -1).sum(dim=-1)
        except NotImplementedError:
            entropy = torch.zeros_like(objective)
        policy_loss = -torch.mean(discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1]))
        fabric.backward(policy_loss)
        actor_grads = None
        if cfg.algo.actor.clip_gradients is not None and cfg.algo.actor.clip_gradients > 0:
            actor_grads = fabric.clip_gradients(
                module=actor, optimizer=actor_optimizer, max_norm=cfg.algo.actor.clip_gradients, error_if_nonfinite=False
            )
        actor_optimizer.step()

        # Predict the values
        qv = TwoHotEncodingDistribution(critic(imagined_trajectories.detach()[:-1]), dims=1)
        predicted_target_values = TwoHotEncodingDistribution(
            target_critic(imagined_trajectories.detach()[:-1]), dims=1
        ).mean

        # Critic optimization. Eq. 10 in the paper
        critic_optimizer.zero_grad(set_to_none=True)
        value_loss = -qv.log_prob(lambda_values.detach())
        value_loss = value_loss - qv.log_prob(predicted_target_values.detach())
        value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))

        fabric.backward(value_loss)
        critic_grads = None
        if cfg.algo.critic.clip_gradients is not None and cfg.algo.critic.clip_gradients > 0:
            critic_grads = fabric.clip_gradients(
                module=critic,
                optimizer=critic_optimizer,
                max_norm=cfg.algo.critic.clip_gradients,
                error_if_nonfinite=False,
            )
        critic_optimizer.step()

    # Log metrics
    if aggregator and not aggregator.disabled:
        aggregator.update("Loss/world_model_loss", rec_loss.detach())
        aggregator.update("Loss/observation_loss", observation_loss.detach())
        aggregator.update("Loss/reward_loss", reward_loss.detach())
        aggregator.update("Loss/state_loss", state_loss.detach())
        aggregator.update("Loss/continue_loss", continue_loss.detach())
        if cem_data:
            aggregator.update("Loss/concept_loss", loss_dict['concept_loss'].detach())
            aggregator.update("Loss/orthognality_loss", loss_dict['concept_loss'].detach())
            try:
                aggregator.update("Loss/per_concept_loss", loss_dict['loss_per_concept'].detach())
            except Exception as e:
                print("something went wrong with per concept loss: ", e)
        aggregator.update("State/kl", kl.mean().detach())
        aggregator.update(
            "State/post_entropy",
            Independent(OneHotCategorical(logits=posteriors_logits.detach()), 1).entropy().mean().detach(),
        )
        aggregator.update(
            "State/prior_entropy",
            Independent(OneHotCategorical(logits=priors_logits.detach()), 1).entropy().mean().detach(),
        )
        if compiled_behaviour_learning is not None:
            aggregator.update("Loss/policy_loss", policy_loss.detach())
            aggregator.update("Loss/value_loss", value_loss.detach())
        if world_model_grads:
            aggregator.update("Grads/world_model", world_model_grads.mean().detach())
        if compiled_behaviour_learning is not None:
            if actor_grads:
                aggregator.update("Grads/actor", actor_grads.mean().detach())
            if critic_grads:
                aggregator.update("Grads/critic", critic_grads.mean().detach())

    # Reset everything
    world_optimizer.zero_grad(set_to_none=True)
    if compiled_behaviour_learning is not None:
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)




import h5py
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, StackDataset
import numpy as np
import json


class LIBERODataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, 'r')
        self.data = self.file['data']
        self.demo_keys = [k for k in self.data.keys() if k.startswith('demo_')]

        # Get total number of samples
        self.total_samples = self.data.attrs['total']

        # Get environment arguments
        self.env_args = json.loads(self.data.attrs['env_args'])

        # Create sample index mapping
        self.sample_map = []
        for demo_key in self.demo_keys:
            demo = self.data[demo_key]
            num_samples = demo.attrs['num_samples']
            self.sample_map.extend([(demo_key, i) for i in range(num_samples)])

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        demo_key, sample_idx = self.sample_map[idx]
        demo = self.data[demo_key]

        # Load observation
        obs = {k: torch.from_numpy(demo['obs'][k][sample_idx]) for k in demo['obs'].keys()}

        # Load action, reward, and done
        action = torch.from_numpy(demo['actions'][sample_idx])
        reward = torch.tensor(demo['rewards'][sample_idx])
        done = torch.tensor(demo['dones'][sample_idx])

        # Load next observation
        next_obs = {k: torch.from_numpy(demo['next_obs'][k][sample_idx]) for k in demo['next_obs'].keys()}

        return obs, action, reward, done, next_obs

    def close(self):
        self.file.close()

def load_dataset(data_path, batch_size, num_workers=4):
    dataset = LIBERODataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader, dataset.env_args

import timeit
from pathlib import Path
# from torchvision import transforms
from torchvision.transforms import v2
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.datasets import (GroupedTaskDataset, SequenceVLDataset, get_dataset)
from libero.lifelong.utils import (get_task_embs, safe_device, create_experiment_dir)

def get_datasets_from_benchmark(benchmark,libero_folder,seq_len=64,obs_modality=None):
    datasets = []
    descriptions = []
    task_concepts = []
    shape_meta = None
    # n_tasks = benchmark.n_tasks
    n_tasks = benchmark.n_tasks
    seq_len = 64
    if obs_modality is None:
        obs_modality = {'rgb': ['agentview_rgb', 'eye_in_hand_rgb'], 'depth': [], 'low_dim': ['gripper_states', 'joint_states']}
    for i in range(n_tasks):
        # currently we assume tasks from same benchmark have the same shape_meta
        task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(libero_folder, benchmark.get_task_demonstration(i)),
                obs_modality=obs_modality, #cfg.data.obs.modality,
                initialize_obs_utils=(i==0),
                seq_len=seq_len,
                dataset_keys=["actions","dones","rewards"],  #"states"
            # Question: does this truncate or simply segment? if segment, how do you know if sample is end?
            # Answer: you don't, in many of these tasks it seems like they don't care, weirdly
        )
        # if no bddl file loaded in advance
        task_str = benchmark.get_task_demonstration(i)[:-len('_demo.hdf5')]
        bddl_folder = Path(get_libero_path("bddl_files"))  # TODO note this has a bug in it that pathlib is just smart enough to handle
        sample_bddl_filepath = bddl_folder / (task_str + '.bddl')
        concepts_i = get_bddl_concepts(sample_bddl_filepath)
        concepts_dataset_i = TargetDataset(concepts_i, seq_len, len(task_i_dataset))
        task_concepts.append(concepts_dataset_i)
        # add language to the vision dataset, hence we call vl_dataset
        descriptions.append(benchmark.get_task(i).language)
        datasets.append(task_i_dataset)
    return datasets, descriptions, task_concepts


class TargetDataset(Dataset):
    def __init__(self, target_list, sequence_length, num_samples):
        # Repeat the target_list to match the sequence length
        self.targets = np.tile(target_list, (sequence_length, 1))
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.targets  # torch.tensor(self.targets, dtype=torch.float32)


class CombinedDictDataset(Dataset):
    def __init__(self, sequence_dataset, target_dataset):
        self.sequence_dataset = sequence_dataset
        self.target_dataset = target_dataset
        assert len(sequence_dataset) == len(target_dataset), "Datasets must have the same length"

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        sequence_item = self.sequence_dataset[idx]
        target_item = self.target_dataset[idx]

        # Assuming sequence_item is a dictionary
        combined_item = sequence_item.copy()  # Create a copy of the original dictionary
        combined_item['targets'] = target_item  # Add the target as a new key

        return combined_item


# Usage example:
# data_loader, env_args = load_dataset('path/to/your/libero_dataset.hdf5', batch_size=32)

class RestructureAndResizeTransform:
    def __init__(self, image_size, num_samples):
        self.image_size = image_size
        self.num_samples = num_samples
        self.transform = v2.Resize((image_size, image_size))
        # transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.v2.Resize((image_size, image_size)),
            # transforms.ToTensor(),
        # ])
        # self.resize = lambda data: transforms.v2.functional.resize(data['obs']['agentview_rgb'],size=(64,64))

    def __call__(self, batch):
        # Restructure and resize observations
        print(batch[0].keys())
        obs, action, reward, done, next_obs = batch

        # Resize observations
        for k in obs.keys():
            if len(obs[k].shape) >= 3:  # Assume image data has 3 dimensions (C, H, W)
                obs[k] = self.transform(obs[k])


        return obs, action, reward, done


        # actions = actions.unsqueeze(0).expand(self.num_samples, -1, -1)
        # rewards = rewards.unsqueeze(0).expand(self.num_samples, -1)

        # # Split dones into truncated and terminated
        # truncated = dones.unsqueeze(0).expand(self.num_samples, -1)
        # terminated = dones.unsqueeze(0).expand(self.num_samples, -1)

        # Expand is_first
        is_first = is_first.unsqueeze(0).expand(self.num_samples, -1)

        return processed_obs, actions, rewards, truncated, terminated, is_first


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any], pretrain_cfg: Dict[str, Any] = None) -> None:
    device = fabric.device
    rank = fabric.global_rank
    world_size = fabric.world_size

    ## Loading
    if cfg.checkpoint.pretrain_ckpt_path is not None:   # Load Pretrained model for finetuning
        loaded_params = True
        fabric.print(f"Starting from pretrained model at : {cfg.checkpoint.pretrain_ckpt_path}")
        state = fabric.load(pathlib.Path(cfg.checkpoint.pretrain_ckpt_path))
        # All the models must be equal to the ones of the exploration phase
        cfg.algo.lmbda = pretrain_cfg.algo.lmbda
        cfg.algo.horizon = pretrain_cfg.algo.horizon
        cfg.algo.layer_norm = pretrain_cfg.algo.layer_norm
        cfg.algo.dense_units = pretrain_cfg.algo.dense_units
        cfg.algo.mlp_layers = pretrain_cfg.algo.mlp_layers
        cfg.algo.dense_act = pretrain_cfg.algo.dense_act
        cfg.algo.cnn_act = pretrain_cfg.algo.cnn_act
        cfg.algo.unimix = pretrain_cfg.algo.unimix
        cfg.algo.hafner_initialization = pretrain_cfg.algo.hafner_initialization
        cfg.algo.world_model = pretrain_cfg.algo.world_model
        cfg.algo.actor = pretrain_cfg.algo.actor
        cfg.algo.critic = pretrain_cfg.algo.critic
        cfg.algo.cnn_keys = pretrain_cfg.algo.cnn_keys
        cfg.algo.mlp_keys = pretrain_cfg.algo.mlp_keys
    elif cfg.checkpoint.resume_from is not None:  # Finetuning that was interrupted for some reason
        loaded_params = True
        fabric.print(f"Resuming training on: {cfg.checkpoint.resume_from}")
        state = fabric.load(pathlib.Path(cfg.checkpoint.resume_from))
    else:
        loaded_params = False


    # These arguments cannot be changed
    cfg.env.frame_stack = -1
    if 2 ** int(np.log2(cfg.env.screen_size)) != cfg.env.screen_size:
        raise ValueError(f"The screen size must be a power of 2, got: {cfg.env.screen_size}")
    # Create Logger. This will create the logger only on the
    # rank-0 process
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    fabric.print(f"Log dir: {log_dir}")

    if cfg.algo.offline is False:

        # Environment setup
        vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
        envs = vectorized_env(
            [
                partial(
                    RestartOnException,
                    make_env(
                        cfg,
                        cfg.seed + rank * cfg.env.num_envs + i,
                        rank * cfg.env.num_envs,
                        log_dir if rank == 0 else None,
                        "train",
                        vector_env_idx=i,
                    ),
                )
                for i in range(cfg.env.num_envs)
            ]
        )
        action_space = envs.single_action_space
        observation_space = envs.single_observation_space

        is_continuous = isinstance(action_space, gym.spaces.Box)
        is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
        actions_dim = tuple(
            action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
        )
        clip_rewards_fn = lambda r: np.tanh(r) if cfg.env.clip_rewards else r
        if not isinstance(observation_space, gym.spaces.Dict):
            raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")

        if (
            len(set(cfg.algo.cnn_keys.encoder).intersection(set(cfg.algo.cnn_keys.decoder))) == 0
            and len(set(cfg.algo.mlp_keys.encoder).intersection(set(cfg.algo.mlp_keys.decoder))) == 0
        ):
            raise RuntimeError("The CNN keys or the MLP keys of the encoder and decoder must not be disjointed")
        if len(set(cfg.algo.cnn_keys.decoder) - set(cfg.algo.cnn_keys.encoder)) > 0:
            raise RuntimeError(
                "The CNN keys of the decoder must be contained in the encoder ones. "
                f"Those keys are decoded without being encoded: {list(set(cfg.algo.cnn_keys.decoder))}"
            )
        if len(set(cfg.algo.mlp_keys.decoder) - set(cfg.algo.mlp_keys.encoder)) > 0:
            raise RuntimeError(
                "The MLP keys of the decoder must be contained in the encoder ones. "
                f"Those keys are decoded without being encoded: {list(set(cfg.algo.mlp_keys.decoder))}"
            )
        if cfg.metric.log_level > 0:
            fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
            fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)
            fabric.print("Decoder CNN keys:", cfg.algo.cnn_keys.decoder)
            fabric.print("Decoder MLP keys:", cfg.algo.mlp_keys.decoder)
        obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder

        # Compile dynamic_learning method
        compiled_dynamic_learning = torch.compile(dynamic_learning, **cfg.algo.compile_dynamic_learning)

        # Compile behaviour_learning method
        compiled_behaviour_learning = torch.compile(behaviour_learning, **cfg.algo.compile_behaviour_learning)

        # Compile compute_lambda_values method
        compiled_compute_lambda_values = torch.compile(compute_lambda_values, **cfg.algo.compile_compute_lambda_values)

        world_model, actor, critic, target_critic, player = build_agent(
            fabric,
            actions_dim,
            is_continuous,
            cfg,
            observation_space,
            state["world_model"] if loaded_params else None,
            state["actor"] if loaded_params else None,
            state["critic"] if loaded_params else None,
            state["target_critic"] if loaded_params else None,
        )

        # Optimizers
        world_optimizer = hydra.utils.instantiate(
            cfg.algo.world_model.optimizer, params=world_model.parameters(), _convert_="all"
        )
        actor_optimizer = hydra.utils.instantiate(cfg.algo.actor.optimizer, params=actor.parameters(), _convert_="all")
        critic_optimizer = hydra.utils.instantiate(cfg.algo.critic.optimizer, params=critic.parameters(), _convert_="all")
        if loaded_params:
            world_optimizer.load_state_dict(state["world_optimizer"])
            actor_optimizer.load_state_dict(state["actor_optimizer"])
            critic_optimizer.load_state_dict(state["critic_optimizer"])
        world_optimizer, actor_optimizer, critic_optimizer = fabric.setup_optimizers(
            world_optimizer, actor_optimizer, critic_optimizer
        )
        moments = Moments(
            cfg.algo.actor.moments.decay,
            cfg.algo.actor.moments.max,
            cfg.algo.actor.moments.percentile.low,
            cfg.algo.actor.moments.percentile.high,
        )
        if loaded_params:
            moments.load_state_dict(state["moments"])

        if fabric.is_global_zero:
            save_configs(cfg, log_dir)

        # Metrics
        aggregator = None
        if not MetricAggregator.disabled:
            aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

        # Local data
        buffer_size = cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size) if not cfg.dry_run else 2
        rb = EnvIndependentReplayBuffer(
            buffer_size,
            n_envs=cfg.env.num_envs,
            memmap=cfg.buffer.memmap,
            memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
            buffer_cls=SequentialReplayBuffer,
        )
        if cfg.checkpoint.resume_from and cfg.buffer.checkpoint or (cfg.buffer.load_from_exploration and cfg.checkpoint.buffer.load_from_pretrain):
            if isinstance(state["rb"], list) and fabric.world_size == len(state["rb"]):
                rb = state["rb"][fabric.global_rank]
            elif isinstance(state["rb"], EnvIndependentReplayBuffer):
                rb = state["rb"]
            else:
                raise RuntimeError(f"Given {len(state['rb'])}, but {fabric.world_size} processes are instantiated")

        # Global variables
        train_step = 0
        last_train = 0
        start_iter = (
            # + 1 because the checkpoint is at the end of the update step
            # (when resuming from a checkpoint, the update at the checkpoint
            # is ended and you have to start with the next one)
            (state["iter_num"] // fabric.world_size) + 1
            if cfg.checkpoint.resume_from
            else 1
        )
        policy_step = state["iter_num"] * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
        last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
        last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
        policy_steps_per_iter = int(cfg.env.num_envs * fabric.world_size)
        total_iters = int(cfg.algo.total_steps // policy_steps_per_iter) if not cfg.dry_run else 1
        learning_starts = cfg.algo.learning_starts // policy_steps_per_iter if not cfg.dry_run else 0
        prefill_steps = learning_starts - int(learning_starts > 0)
        if cfg.checkpoint.resume_from:
            cfg.algo.per_rank_batch_size = state["batch_size"] // fabric.world_size
            learning_starts += start_iter
            prefill_steps += start_iter

        # Create Ratio class
        ratio = Ratio(cfg.algo.replay_ratio, pretrain_steps=cfg.algo.per_rank_pretrain_steps)
        if cfg.checkpoint.resume_from:
            ratio.load_state_dict(state["ratio"])

        # Warning for log and checkpoint every
        if cfg.metric.log_level > 0 and cfg.metric.log_every % policy_steps_per_iter != 0:
            warnings.warn(
                f"The metric.log_every parameter ({cfg.metric.log_every}) is not a multiple of the "
                f"policy_steps_per_iter value ({policy_steps_per_iter}), so "
                "the metrics will be logged at the nearest greater multiple of the "
                "policy_steps_per_iter value."
            )
        if cfg.checkpoint.every % policy_steps_per_iter != 0:
            warnings.warn(
                f"The checkpoint.every parameter ({cfg.checkpoint.every}) is not a multiple of the "
                f"policy_steps_per_iter value ({policy_steps_per_iter}), so "
                "the checkpoint will be saved at the nearest greater multiple of the "
                "policy_steps_per_iter value."
            )

        # Get the first environment observation and start the optimization
        step_data = {}
        obs = envs.reset(seed=cfg.seed)[0]
        for k in obs_keys:
            step_data[k] = obs[k][np.newaxis]
        step_data["rewards"] = np.zeros((1, cfg.env.num_envs, 1))
        step_data["truncated"] = np.zeros((1, cfg.env.num_envs, 1))
        step_data["terminated"] = np.zeros((1, cfg.env.num_envs, 1))
        step_data["is_first"] = np.ones_like(step_data["terminated"])
        if cfg.algo.world_model.cbm_model.n_concepts:
            step_data["targets"] = np.zeros((1, cfg.env.num_envs, cfg.algo.world_model.cbm_model.n_concepts))
        player.init_states()
        
        top_ep_rew = []
        last_ep_rew = 0

        cumulative_per_rank_gradient_steps = 0
        for iter_num in range(start_iter, total_iters + 1):
            policy_step += policy_steps_per_iter

            ## Collect Data Phase
            with torch.inference_mode():
                # Measure environment interaction time: this considers both the model forward
                # to get the action given the observation and the time taken into the environment
                with timer("Time/env_interaction_time", SumMetric, sync_on_compute=False):
                    # Sample an action given the observation received by the environment
                    if (
                        iter_num <= learning_starts
                        and cfg.checkpoint.resume_from is None
                        and "minedojo" not in cfg.env.wrapper._target_.lower()
                    ):
                        real_actions = actions = np.array(envs.action_space.sample())
                        if not is_continuous:
                            actions = np.concatenate(
                                [
                                    F.one_hot(torch.as_tensor(act), act_dim).numpy()
                                    for act, act_dim in zip(actions.reshape(len(actions_dim), -1), actions_dim)
                                ],
                                axis=-1,
                            )
                    else:
                        torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs)
                        mask = {k: v for k, v in torch_obs.items() if k.startswith("mask")}
                        if len(mask) == 0:
                            mask = None
                        real_actions = actions = player.get_actions(torch_obs, mask=mask)
                        actions = torch.cat(actions, -1).cpu().numpy()
                        if is_continuous:
                            real_actions = torch.stack(real_actions, dim=-1).cpu().numpy()
                        else:
                            real_actions = (
                                torch.stack([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()
                            )

                    step_data["actions"] = actions.reshape((1, cfg.env.num_envs, -1))
                    rb.add(step_data, validate_args=cfg.buffer.validate_args)

                    next_obs, rewards, terminated, truncated, infos = envs.step(
                        real_actions.reshape(envs.action_space.shape)
                    )
                    dones = np.logical_or(terminated, truncated).astype(np.uint8)

                step_data["is_first"] = np.zeros_like(step_data["terminated"])
                if "restart_on_exception" in infos:
                    for i, agent_roe in enumerate(infos["restart_on_exception"]):
                        if agent_roe and not dones[i]:
                            last_inserted_idx = (rb.buffer[i]._pos - 1) % rb.buffer[i].buffer_size
                            rb.buffer[i]["terminated"][last_inserted_idx] = np.zeros_like(
                                rb.buffer[i]["terminated"][last_inserted_idx]
                            )
                            rb.buffer[i]["truncated"][last_inserted_idx] = np.ones_like(
                                rb.buffer[i]["truncated"][last_inserted_idx]
                            )
                            rb.buffer[i]["is_first"][last_inserted_idx] = np.zeros_like(
                                rb.buffer[i]["is_first"][last_inserted_idx]
                            )
                            step_data["is_first"][i] = np.ones_like(step_data["is_first"][i])

                if cfg.metric.log_level > 0 and "final_info" in infos:
                    for i, agent_ep_info in enumerate(infos["final_info"]):
                        if agent_ep_info is not None:
                            ep_rew = agent_ep_info["episode"]["r"]
                            ep_len = agent_ep_info["episode"]["l"]
                            last_ep_rew = ep_rew.mean()
                            if aggregator and not aggregator.disabled:
                                aggregator.update("Rewards/rew_avg", ep_rew)
                                aggregator.update("Game/ep_len_avg", ep_len)
                            fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew[-1]}")

                # Save the real next observation
                real_next_obs = copy.deepcopy(next_obs)
                if "final_observation" in infos:
                    for idx, final_obs in enumerate(infos["final_observation"]):
                        if final_obs is not None:
                            for k, v in final_obs.items():
                                real_next_obs[k][idx] = v

                for k in obs_keys:
                    step_data[k] = next_obs[k][np.newaxis]

                if "concepts" in infos:
                    step_data["targets"] = np.expand_dims(np.stack(infos["concepts"]),0)

                # next_obs becomes the new obs
                obs = next_obs

                rewards = rewards.reshape((1, cfg.env.num_envs, -1))
                step_data["terminated"] = terminated.reshape((1, cfg.env.num_envs, -1))
                step_data["truncated"] = truncated.reshape((1, cfg.env.num_envs, -1))
                step_data["rewards"] = clip_rewards_fn(rewards)

                dones_idxes = dones.nonzero()[0].tolist()
                reset_envs = len(dones_idxes)
                if reset_envs > 0:
                    reset_data = {}
                    for k in obs_keys:
                        reset_data[k] = (real_next_obs[k][dones_idxes])[np.newaxis]
                    reset_data["terminated"] = step_data["terminated"][:, dones_idxes]
                    reset_data["truncated"] = step_data["truncated"][:, dones_idxes]
                    reset_data["actions"] = np.zeros((1, reset_envs, np.sum(actions_dim)))
                    reset_data["rewards"] = step_data["rewards"][:, dones_idxes]
                    reset_data["is_first"] = np.zeros_like(reset_data["terminated"])
                    rb.add(reset_data, dones_idxes, validate_args=cfg.buffer.validate_args)

                    # Reset already inserted step data
                    step_data["rewards"][:, dones_idxes] = np.zeros_like(reset_data["rewards"])
                    step_data["terminated"][:, dones_idxes] = np.zeros_like(step_data["terminated"][:, dones_idxes])
                    step_data["truncated"][:, dones_idxes] = np.zeros_like(step_data["truncated"][:, dones_idxes])
                    step_data["is_first"][:, dones_idxes] = np.ones_like(step_data["is_first"][:, dones_idxes])
                    player.init_states(dones_idxes)

            ## Train the agent Phase
            if iter_num >= learning_starts:
                ratio_steps = policy_step - prefill_steps * policy_steps_per_iter
                per_rank_gradient_steps = ratio(ratio_steps / world_size)
                if per_rank_gradient_steps > 0:  # Sample data from RB
                    local_data = rb.sample_tensors(
                        cfg.algo.per_rank_batch_size,
                        sequence_length=cfg.algo.per_rank_sequence_length,
                        n_samples=per_rank_gradient_steps,
                        dtype=None,
                        device=fabric.device,
                        from_numpy=cfg.buffer.from_numpy,
                    )
                    with timer("Time/train_time", SumMetric, sync_on_compute=cfg.metric.sync_on_compute):
                        for i in range(per_rank_gradient_steps):
                            if (
                                cumulative_per_rank_gradient_steps % cfg.algo.critic.per_rank_target_network_update_freq
                                == 0
                            ):
                                tau = 1 if cumulative_per_rank_gradient_steps == 0 else cfg.algo.critic.tau
                                for cp, tcp in zip(critic.module.parameters(), target_critic.parameters()):
                                    tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)
                            batch = {k: v[i].float() for k, v in local_data.items()}
                            train(
                                fabric=fabric,
                                world_model=world_model,
                                actor=actor,
                                critic=critic,
                                target_critic=target_critic,
                                world_optimizer=world_optimizer,
                                actor_optimizer=actor_optimizer,
                                critic_optimizer=critic_optimizer,
                                data=batch,
                                aggregator=aggregator,
                                cfg=cfg,
                                is_continuous=is_continuous,
                                actions_dim=actions_dim,
                                moments=moments,
                                compiled_dynamic_learning=compiled_dynamic_learning,
                                compiled_behaviour_learning=compiled_behaviour_learning,
                                compiled_compute_lambda_values=compiled_compute_lambda_values,
                            )
                            cumulative_per_rank_gradient_steps += 1
                        train_step += world_size

            ## Log metrics Phase
            if cfg.metric.log_level > 0 and (policy_step - last_log >= cfg.metric.log_every or iter_num == total_iters):
                # Get envrionment metrics
                if cfg.env.env_stats:
                    envs_stats = {}
                    for env in envs.envs:
                        env_stats = env.get_env_stats(
                            stats_dict=dict(cfg.env.env_stats),
                            reset=True)
                        if env_stats is None or len(list(env_stats.values())[0]) == 0:
                            # print('not updating, no new data')
                            break
                        if not envs_stats:
                            envs_stats = env_stats
                        else:
                            for k, v in env_stats.items():
                                envs_stats[k].extend(v)
                    else:  # For-else statement only if no break occurs
                        for k, v in envs_stats.items():
                            if len(v) == 0:
                                # print('not updating, no new data (for else)')
                                break
                            aggregator.update(k, np.array(v).reshape(1,cfg.env.num_envs,-1))
                            # fabric.log(f"Env/{k}", v, policy_step)

                # Sync distributed metrics
                if aggregator and not aggregator.disabled:
                    metrics_dict = aggregator.compute()
                    fabric.log_dict(metrics_dict, policy_step)
                    aggregator.reset()

                # Log replay ratio
                fabric.log(
                    "Params/replay_ratio", cumulative_per_rank_gradient_steps * world_size / policy_step, policy_step
                )


                ## TODO Testing: how to save images when you have wandb
                # if "wandb" in cfg.metric.logger._target_.lower():
                #     fabric.logger.log_image(key="samples", images=[batch['agentview_rgb'][0,0,...]])

                # Sync distributed timers
                if not timer.disabled:
                    timer_metrics = timer.compute()
                    if "Time/train_time" in timer_metrics and timer_metrics["Time/train_time"] > 0:
                        fabric.log(
                            "Time/sps_train",
                            (train_step - last_train) / timer_metrics["Time/train_time"],
                            policy_step,
                        )
                    if "Time/env_interaction_time" in timer_metrics and timer_metrics["Time/env_interaction_time"] > 0:
                        fabric.log(
                            "Time/sps_env_interaction",
                            ((policy_step - last_log) / world_size * cfg.env.action_repeat)
                            / timer_metrics["Time/env_interaction_time"],
                            policy_step,
                        )
                    timer.reset()

                # Reset counters
                last_log = policy_step
                last_train = train_step

            ## Checkpoint Model Phase
            if (cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every) or (
                iter_num == total_iters and cfg.checkpoint.save_last
            ):
                if (len(top_ep_rew) < cfg.checkpoint.keep_last) or (last_ep_rew > min(top_ep_rew)):
                    if len(top_ep_rew) >= cfg.checkpoint.keep_last:
                        top_ep_rew.remove(min(top_ep_rew))
                    top_ep_rew.append(last_ep_rew)
                    
                    last_checkpoint = policy_step
                    state = {
                        "world_model": world_model.state_dict(),
                        "actor": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "target_critic": target_critic.state_dict(),
                        "world_optimizer": world_optimizer.state_dict(),
                        "actor_optimizer": actor_optimizer.state_dict(),
                        "critic_optimizer": critic_optimizer.state_dict(),
                        "moments": moments.state_dict(),
                        "ratio": ratio.state_dict(),
                        "iter_num": iter_num * fabric.world_size,
                        "batch_size": cfg.algo.per_rank_batch_size * fabric.world_size,
                        "last_log": last_log,
                        "last_checkpoint": last_checkpoint,
                    }
                    ckpt_path = log_dir + f"/checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt"
                    fabric.call(
                        "on_checkpoint_coupled",
                        fabric=fabric,
                        ckpt_path=ckpt_path,
                        state=state,
                        replay_buffer=rb if cfg.buffer.checkpoint else None,
                    )

        envs.close()
    else:  ## OFFLINE LEARNING

        ####Comment out
        ## Environment setup
        vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
        envs = vectorized_env(
            [
                partial(
                    RestartOnException,
                    make_env(
                        cfg,
                        cfg.seed + rank * cfg.env.num_envs + i,
                        rank * cfg.env.num_envs,
                        log_dir if rank == 0 else None,
                        "train",
                        vector_env_idx=i,
                    ),
                )
                for i in range(cfg.env.num_envs)
            ]
        )
        ##### END COMMENT OUT


        libero_folder = get_libero_path("datasets")
        bddl_folder = get_libero_path("bddl_files")
        init_states_folder = get_libero_path("init_states")
        eval_num_procs = 1
        eval_n_eval = 5

        train_n_epochs = 25

        task_order = 0  # Default task order
        benchmark_name = "libero_90" # TODO add to config can be from {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
        benchmark = get_benchmark(benchmark_name)(task_order)
        obs_modality = {'rgb': ['agentview_rgb']} #, 'low_dim': [ 'joint_states']}

        datasets, descriptions, task_concepts = get_datasets_from_benchmark(
            benchmark=benchmark,
            libero_folder=libero_folder,
            seq_len=cfg.algo.per_rank_sequence_length,
            obs_modality=obs_modality
            )
        # task_embs = get_task_embs(cfg,descriptions)
        # benchmark.set_task_embs(task_embs)
        n_demos = [data.n_demos for data in datasets]
        n_sequences = [data.total_num_sequences for data in datasets]
        # dataloader = DataLoader(
        #     dataset,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     num_workers=num_workers,
        #     pin_memory=True
        # )

        # Create Ratio class
        ratio = Ratio(cfg.algo.replay_ratio, pretrain_steps=cfg.algo.per_rank_pretrain_steps)
        if cfg.checkpoint.resume_from:
            ratio.load_state_dict(state["ratio"])

        if cfg.algo.offline_supervision:
            temp_datas = []
            for dataset, task_concept in zip(datasets, task_concepts):
                temp_datas.append(CombinedDictDataset(dataset, task_concept))
                # temp_datas.append(StackDataset(dataset, task_concept))
            datasets = temp_datas
        concat_dataset = ConcatDataset(datasets)
        action_sample = concat_dataset[0]['actions']
        if isinstance(action_sample, np.ndarray):
            action_space = gym.spaces.Box(
                low=-1.0, #np.min(action_sample),
                high=1.0, #np.max(action_sample),
                shape=action_sample[0].shape,
                dtype=action_sample.dtype
            )
        obs_sample = concat_dataset[0]['obs']  # Assuming dataset[i][0] is the observation
        if isinstance(obs_sample, dict):
            obs_space_dict = {}
            for k, v in obs_sample.items():
                if 'image' in k or 'rgb' in k:
                    obs_space_dict[k] = gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=v[0].shape,
                        dtype=v.dtype
                    )
                else:
                    obs_space_dict[k] = gym.spaces.Box(
                        low=-1.0,
                        high=1.0,
                        shape=v[0].shape,
                        dtype=v.dtype
                    )
            observation_space = gym.spaces.Dict(obs_space_dict)
        ## NOTE: all Libero90 datapoints are concatonated sequences of 64 frames.


        # transform = RestructureAndResizeTransform(
        #     cfg.env.screen_size,
        #     num_samples=2)


        dataloader = DataLoader(
            concat_dataset,
            batch_size=cfg.algo.per_rank_batch_size * 2, # replay_ratio=0.5 This is hacky, but Ratio is confusing
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            # collate_fn=transform,
            )


        action_space = envs.single_action_space
        # action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        observation_space = envs.single_observation_space
        # observation_space = gym.spaces.Dict({
        #     'rgb': gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
        #     'state': gym.spaces.Box(low=-1, high=1, shape=(32,), dtype=np.float64)
        # })

        is_continuous = isinstance(action_space, gym.spaces.Box)
        is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
        actions_dim = tuple(
            action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
        )
        clip_rewards_fn = lambda r: np.tanh(r) if cfg.env.clip_rewards else r
        if not isinstance(observation_space, gym.spaces.Dict):
            raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")

        if (
            len(set(cfg.algo.cnn_keys.encoder).intersection(set(cfg.algo.cnn_keys.decoder))) == 0
            and len(set(cfg.algo.mlp_keys.encoder).intersection(set(cfg.algo.mlp_keys.decoder))) == 0
        ):
            raise RuntimeError("The CNN keys or the MLP keys of the encoder and decoder must not be disjointed")
        if len(set(cfg.algo.cnn_keys.decoder) - set(cfg.algo.cnn_keys.encoder)) > 0:
            raise RuntimeError(
                "The CNN keys of the decoder must be contained in the encoder ones. "
                f"Those keys are decoded without being encoded: {list(set(cfg.algo.cnn_keys.decoder))}"
            )
        if len(set(cfg.algo.mlp_keys.decoder) - set(cfg.algo.mlp_keys.encoder)) > 0:
            raise RuntimeError(
                "The MLP keys of the decoder must be contained in the encoder ones. "
                f"Those keys are decoded without being encoded: {list(set(cfg.algo.mlp_keys.decoder))}"
            )
        if cfg.metric.log_level > 0:
            fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
            fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)
            fabric.print("Decoder CNN keys:", cfg.algo.cnn_keys.decoder)
            fabric.print("Decoder MLP keys:", cfg.algo.mlp_keys.decoder)
        obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder

        # Compile dynamic_learning method
        compiled_dynamic_learning = torch.compile(dynamic_learning, **cfg.algo.compile_dynamic_learning)

        # Compile behaviour_learning method
        compiled_behaviour_learning = torch.compile(behaviour_learning, **cfg.algo.compile_behaviour_learning)

        # Compile compute_lambda_values method
        compiled_compute_lambda_values = torch.compile(compute_lambda_values, **cfg.algo.compile_compute_lambda_values)

        world_model, actor, critic, target_critic, player = build_agent(
            fabric,
            actions_dim,
            is_continuous,
            cfg,
            observation_space,
            state["world_model"] if loaded_params else None,
            state["actor"] if loaded_params else None,
            state["critic"] if loaded_params else None,
            state["target_critic"] if loaded_params else None,
        )

        # Optimizers
        world_optimizer = hydra.utils.instantiate(
            cfg.algo.world_model.optimizer, params=world_model.parameters(), _convert_="all"
        )
        actor_optimizer = hydra.utils.instantiate(cfg.algo.actor.optimizer, params=actor.parameters(), _convert_="all")
        critic_optimizer = hydra.utils.instantiate(cfg.algo.critic.optimizer, params=critic.parameters(), _convert_="all")
        if loaded_params:
            world_optimizer.load_state_dict(state["world_optimizer"])
            actor_optimizer.load_state_dict(state["actor_optimizer"])
            critic_optimizer.load_state_dict(state["critic_optimizer"])
        world_optimizer, actor_optimizer, critic_optimizer = fabric.setup_optimizers(
            world_optimizer, actor_optimizer, critic_optimizer
        )
        moments = Moments(
            cfg.algo.actor.moments.decay,
            cfg.algo.actor.moments.max,
            cfg.algo.actor.moments.percentile.low,
            cfg.algo.actor.moments.percentile.high,
        )
        if loaded_params:
            moments.load_state_dict(state["moments"])

        if fabric.is_global_zero:
            save_configs(cfg, log_dir)

        # Metrics
        aggregator = None
        if not MetricAggregator.disabled:
            aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

        # Local data
        buffer_size = cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size) if not cfg.dry_run else 2
        rb = EnvIndependentReplayBuffer(
            buffer_size,
            n_envs=cfg.env.num_envs,
            memmap=cfg.buffer.memmap,
            memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
            buffer_cls=SequentialReplayBuffer,
        )

        if cfg.checkpoint.resume_from and cfg.buffer.checkpoint:
            if isinstance(state["rb"], list) and fabric.world_size == len(state["rb"]):
                rb = state["rb"][fabric.global_rank]
            elif isinstance(state["rb"], EnvIndependentReplayBuffer):
                rb = state["rb"]
            else:
                raise RuntimeError(f"Given {len(state['rb'])}, but {fabric.world_size} processes are instantiated")

        # Global variables
        train_step = 0
        last_train = 0
        start_iter = (
            # + 1 because the checkpoint is at the end of the update step
            # (when resuming from a checkpoint, the update at the checkpoint
            # is ended and you have to start with the next one)
            (state["iter_num"] // fabric.world_size) + 1
            if cfg.checkpoint.resume_from
            else 1
        )
        policy_step = state["iter_num"] * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
        last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
        last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
        policy_steps_per_iter = int(cfg.env.num_envs * fabric.world_size)
        total_iters = int(cfg.algo.total_steps // policy_steps_per_iter) if not cfg.dry_run else 1
        learning_starts = cfg.algo.learning_starts // policy_steps_per_iter if not cfg.dry_run else 0
        prefill_steps = learning_starts - int(learning_starts > 0)
        if cfg.checkpoint.resume_from:
            cfg.algo.per_rank_batch_size = state["batch_size"] // fabric.world_size
            learning_starts += start_iter
            prefill_steps += start_iter

        # Warning for log and checkpoint every
        if cfg.metric.log_level > 0 and cfg.metric.log_every % policy_steps_per_iter != 0:
            warnings.warn(
                f"The metric.log_every parameter ({cfg.metric.log_every}) is not a multiple of the "
                f"policy_steps_per_iter value ({policy_steps_per_iter}), so "
                "the metrics will be logged at the nearest greater multiple of the "
                "policy_steps_per_iter value."
            )
        if cfg.checkpoint.every % policy_steps_per_iter != 0:
            warnings.warn(
                f"The checkpoint.every parameter ({cfg.checkpoint.every}) is not a multiple of the "
                f"policy_steps_per_iter value ({policy_steps_per_iter}), so "
                "the checkpoint will be saved at the nearest greater multiple of the "
                "policy_steps_per_iter value."
            )

        # Get the first environment observation and start the optimization
        step_data = {}
        obs = envs.reset(seed=cfg.seed)[0]
        for k in obs_keys:
            step_data[k] = obs[k][np.newaxis]
        step_data["rewards"] = np.zeros((1, cfg.env.num_envs, 1))
        step_data["truncated"] = np.zeros((1, cfg.env.num_envs, 1))
        step_data["terminated"] = np.zeros((1, cfg.env.num_envs, 1))
        step_data["is_first"] = np.ones_like(step_data["terminated"])
        player.init_states()

        cumulative_per_rank_gradient_steps = 0
        iter_num = start_iter
        num_epochs = 100

        init_batch = next(iter(dataloader))

        is_first_dummy_tensor = torch.cat((
            torch.ones((1,init_batch['dones'].shape[0])),
            torch.zeros((init_batch['dones'].shape[1]-1,init_batch['dones'].shape[0]))
        )).T.contiguous()  # hack. this only works when the sequence length is the same for all, 64 in the case of libero_90.
        is_first_dummy_tensor = is_first_dummy_tensor.view(
            2,
            is_first_dummy_tensor.shape[0]//2,
            *is_first_dummy_tensor.shape[1:]).permute(0,2,1,).unsqueeze(-1).to(device)

        #             torch.ones_like(batch['dones'].shape[0]),
        #             torch.zeros_like(batch['dones'].shape[0])
        # for iter_num in range(start_iter, total_iters + 1):

        cfg.buffer.checkpoint = False  #only when offline learning TODO probably should be an assert

        learning_starts = 64
        for epoch in range(num_epochs):
            print(f"Epoch {epoch}")
            for batch in dataloader:
                print(f"iter_num={iter_num}")
                ## Transform data to make it match env. TODO This feels like I should be able to do it in the dataloader:
                batch['truncated'] = batch['dones']
                batch['terminated'] = batch['dones']
                for obskey, obs_tmp in batch['obs'].items():
                    if 'rgb' in obskey and obs_tmp.shape[-2] != cfg.env.screen_size:
                        # obs_tmp = F.interpolate(obs_tmp, (obs_tmp.shape[-3],cfg.env.screen_size,cfg.env.screen_size),mode='nearest-exact') # NOTE: this is slightly better 0.1s
                        obs_tmp = v2.functional.resize(obs_tmp, (cfg.env.screen_size,cfg.env.screen_size)) # NOTE: this is slow, 0.3s
                    batch[obskey] = obs_tmp

                for key, v in batch.items():
                    # expand batch to "2 samplings"
                    if isinstance(v, torch.Tensor):
                        batch[key] = v.view(
                            2,
                            v.shape[0]//2,
                            *v.shape[1:]).to(device) # NOTE: moving image data to GPU takes about 0.03s, can it be faster?
                        # permute to match env
                        if len(batch[key].shape) == 3:
                            batch[key] = batch[key].permute(0,2,1,).unsqueeze(-1)
                        elif len(batch[key].shape) == 4:
                            batch[key] = batch[key].permute(0,2,1,3)
                        elif len(batch[key].shape) == 6:  # rgb
                            batch[key] = batch[key].permute(0,2,1,*range(3,len(batch[key].shape)))  #4,5,3)  # *range(3,len(v.shape)))
                        else:
                            raise NotImplementedError(
                                f"All shapes should be 3,4, or 6D, got {len(batch[key].shape)} for {key}")
                batch['is_first'] = is_first_dummy_tensor
                # batch['agentview_rgb'] = batch['obs']['agentview_rgb'].permute(1,0,3,4,2)
                del batch['obs']
                del batch['dones']

                policy_step += policy_steps_per_iter

                ratio_steps = policy_step - prefill_steps * policy_steps_per_iter
                per_rank_gradient_steps = ratio(ratio_steps / world_size)
                if per_rank_gradient_steps > 0:  # Sample data from RB
                    # local_data = rb.sample_tensors(
                    #     cfg.algo.per_rank_batch_size,
                    #     sequence_length=cfg.algo.per_rank_sequence_length,
                    #     n_samples=per_rank_gradient_steps,
                    #     dtype=None,
                    #     device=fabric.device,
                    #     from_numpy=cfg.buffer.from_numpy,
                    # )
                    with timer("Time/train_time", SumMetric, sync_on_compute=cfg.metric.sync_on_compute):
                        for i in range(per_rank_gradient_steps):
                            if (
                                cumulative_per_rank_gradient_steps % cfg.algo.critic.per_rank_target_network_update_freq
                                == 0
                            ):
                                tau = 1 if cumulative_per_rank_gradient_steps == 0 else cfg.algo.critic.tau
                                for cp, tcp in zip(critic.module.parameters(), target_critic.parameters()):
                                    tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)


                            # shaped_local_data = {k: v[i].float() for k, v in local_data.items()}
                            shaped_batch = {k: v[i].float() for k, v in batch.items()}
                            train(
                                fabric=fabric,
                                world_model=world_model,
                                actor=actor,
                                critic=critic,
                                target_critic=target_critic,
                                world_optimizer=world_optimizer,
                                actor_optimizer=actor_optimizer,
                                critic_optimizer=critic_optimizer,
                                data=shaped_batch,
                                aggregator=aggregator,
                                cfg=cfg,
                                is_continuous=is_continuous,
                                actions_dim=actions_dim,
                                moments=moments,
                                compiled_dynamic_learning=compiled_dynamic_learning,
                                compiled_behaviour_learning=None,
                                compiled_compute_lambda_values=compiled_compute_lambda_values,
                            )
                            cumulative_per_rank_gradient_steps += 1
                        train_step += world_size

                ## Log metrics Phase
                if cfg.metric.log_level > 0 and (policy_step - last_log >= cfg.metric.log_every or iter_num == total_iters):
                    print(f"policy_step={policy_step}, last_log={last_log}")
                    # Sync distributed metrics
                    if aggregator and not aggregator.disabled:
                        metrics_dict = aggregator.compute()
                        fabric.log_dict(metrics_dict, policy_step)
                        aggregator.reset()

                    # Log replay ratio
                    fabric.log(
                        "Params/replay_ratio", cumulative_per_rank_gradient_steps * world_size / policy_step, policy_step
                    )

                    # Sync distributed timers
                    if not timer.disabled:
                        timer_metrics = timer.compute()
                        if "Time/train_time" in timer_metrics and timer_metrics["Time/train_time"] > 0:
                            fabric.log(
                                "Time/sps_train",
                                (train_step - last_train) / timer_metrics["Time/train_time"],
                                policy_step,
                            )
                        if "Time/env_interaction_time" in timer_metrics and timer_metrics["Time/env_interaction_time"] > 0:
                            fabric.log(
                                "Time/sps_env_interaction",
                                ((policy_step - last_log) / world_size * cfg.env.action_repeat)
                                / timer_metrics["Time/env_interaction_time"],
                                policy_step,
                            )
                        timer.reset()

                    # Reset counters
                    last_log = policy_step
                    last_train = train_step

                ## Checkpoint Model Phase
                # print(f"policy_step={policy_step}, last_checkpoint={last_checkpoint}, iter_num={iter_num}")
                # print(f" checkpoint? {policy_step - last_checkpoint >= cfg.checkpoint.every}")
                if (cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every) or (
                    iter_num == total_iters and cfg.checkpoint.save_last
                ):                    
                    print(f"checkpoint at policy_step={policy_step}")
                    last_checkpoint = policy_step
                    state = {
                        "world_model": world_model.state_dict(),
                        "actor": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "target_critic": target_critic.state_dict(),
                        "world_optimizer": world_optimizer.state_dict(),
                        "actor_optimizer": actor_optimizer.state_dict(),
                        "critic_optimizer": critic_optimizer.state_dict(),
                        "moments": moments.state_dict(),
                        "ratio": ratio.state_dict(),
                        "iter_num": iter_num * fabric.world_size,
                        "batch_size": cfg.algo.per_rank_batch_size * fabric.world_size,
                        "last_log": last_log,
                        "last_checkpoint": last_checkpoint,
                    }
                    ckpt_path = log_dir + f"/checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt"
                    fabric.call(
                        "on_checkpoint_coupled",
                        fabric=fabric,
                        ckpt_path=ckpt_path,
                        state=state,
                        replay_buffer=rb if cfg.buffer.checkpoint else None,
                    )
                    # print(f"iter_num={iter_num}")
                iter_num += 1  # 1 for num_updates, update_samples = cfg.algo.per_rank_batch_size * fabric.world_size

                if iter_num > total_iters:
                    break
            else:  # Continue if the inner loop wasn't broken
                continue
            # break

    if fabric.is_global_zero and cfg.algo.run_test:
        test(player, fabric, cfg, log_dir, greedy=False)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.dreamer_v1.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {
            "world_model": world_model,
            "actor": actor,
            "critic": critic,
            "target_critic": target_critic,
            "moments": moments,
        }
        register_model(fabric, log_models, cfg, models_to_log)
