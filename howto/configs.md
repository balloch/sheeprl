# Configurations

This document explains how the configuration files and folders are structured. It will help you to understand how to use add new configuration files and where to put them.

> [!WARNING]
>
> Configuration files heavily depend on the `hydra` library. If you are not familiar with `hydra`, you are strongly advised to read their [documentation](https://hydra.cc/docs/intro/) before using this library.

> [!WARNING]
>
> For every possible hydra config, the parameters that are not specified in the config is highly probable that are passed to the object to be instantiated at runtime. If this is not the case, please let us know!

## Parent Folder Structure
```tree
sheeprl/configs
├── algo
│   ├── a2c.yaml
│   ├── default.yaml
│   ├── dreamer_v1.yaml
│   ├── dreamer_v2.yaml
│   ├── dreamer_v3_L.yaml
│   ├── dreamer_v3_M.yaml
│   ├── dreamer_v3_S.yaml
│   ├── dreamer_v3_XL.yaml
│   ├── dreamer_v3_XS.yaml
│   ├── dreamer_v3.yaml
│   ├── offline_dreamer.yaml
│   ├── droq.yaml
│   ├── p2e_dv1.yaml
│   ├── p2e_dv2.yaml
│   ├── p2e_dv3.yaml
│   ├── ppo_decoupled.yaml
│   ├── ppo_recurrent.yaml
│   ├── ppo.yaml
│   ├── sac_ae.yaml
│   ├── sac_decoupled.yaml
│   └── sac.yaml
├── buffer
│   └── default.yaml
├── checkpoint
│   └── default.yaml
├── config.yaml
├── distribution
│   └── default.yaml
├── env
│   ├── atari.yaml
│   ├── crafter.yaml
│   ├── default.yaml
│   ├── diambra.yaml
│   ├── dmc.yaml
│   ├── dummy.yaml
│   ├── gym.yaml
│   ├── minecraft.yaml
│   ├── minedojo.yaml
│   ├── minerl_obtain_diamond.yaml
│   ├── minerl_obtain_iron_pickaxe.yaml
│   ├── minerl.yaml
│   ├── mujoco.yaml
│   └── super_mario_bros.yaml
├── env_config.yaml
├── eval_config.yaml
├── exp
│   ├── a2c_benchmarks.yaml
│   ├── a2c.yaml
│   ├── default.yaml
│   ├── dreamer_v1_benchmarks.yaml
│   ├── dreamer_v1.yaml
│   ├── dreamer_v2_benchmarks.yaml
│   ├── dreamer_v2_crafter.yaml
│   ├── dreamer_v2_ms_pacman.yaml
│   ├── dreamer_v2.yaml
│   ├── dreamer_v3_100k_boxing.yaml
│   ├── dreamer_v3_100k_ms_pacman.yaml
│   ├── dreamer_v3_benchmarks.yaml
│   ├── dreamer_v3_dmc_cartpole_swingup_sparse.yaml
│   ├── dreamer_v3_dmc_walker_walk.yaml
│   ├── dreamer_v3_L_doapp_128px_gray_combo_discrete.yaml
│   ├── dreamer_v3_L_doapp.yaml
│   ├── dreamer_v3_L_navigate.yaml
│   ├── dreamer_v3_super_mario_bros.yaml
│   ├── dreamer_v3_XL_crafter.yaml
│   ├── dreamer_v3.yaml
│   ├── offline_dreamer.yaml
│   ├── droq.yaml
│   ├── p2e_dv1_exploration.yaml
│   ├── p2e_dv1_finetuning.yaml
│   ├── p2e_dv2_exploration.yaml
│   ├── p2e_dv2_finetuning.yaml
│   ├── p2e_dv3_expl_L_doapp_128px_gray_combo_discrete_15Mexpl_20Mstps.yaml
│   ├── p2e_dv3_exploration.yaml
│   ├── p2e_dv3_finetuning.yaml
│   ├── p2e_dv3_fntn_L_doapp_64px_gray_combo_discrete_5Mstps.yaml
│   ├── ppo_benchmarks.yaml
│   ├── ppo_decoupled.yaml
│   ├── ppo_recurrent.yaml
│   ├── ppo_super_mario_bros.yaml
│   ├── ppo.yaml
│   ├── sac_ae.yaml
│   ├── sac_benchmarks.yaml
│   ├── sac_decoupled.yaml
│   └── sac.yaml
├── fabric
│   ├── ddp-cpu.yaml
│   ├── ddp-cuda.yaml
│   └── default.yaml
├── hydra
│   └── default.yaml
├── __init__.py
├── logger
│   ├── mlflow.yaml
│   └── tensorboard.yaml
├── metric
│   └── default.yaml
├── model_manager
│   ├── a2c.yaml
│   ├── default.yaml
│   ├── dreamer_v1.yaml
│   ├── dreamer_v2.yaml
│   ├── dreamer_v3.yaml
│   ├── offline_dreamer.yaml
│   ├── droq.yaml
│   ├── p2e_dv1_exploration.yaml
│   ├── p2e_dv1_finetuning.yaml
│   ├── p2e_dv2_exploration.yaml
│   ├── p2e_dv2_finetuning.yaml
│   ├── p2e_dv3_exploration.yaml
│   ├── p2e_dv3_finetuning.yaml
│   ├── ppo_recurrent.yaml
│   ├── ppo.yaml
│   ├── sac_ae.yaml
│   └── sac.yaml
├── model_manager_config.yaml
└── optim
    ├── adam.yaml
    ├── rmsprop_tf.yaml
    ├── rmsprop.yaml
    └── sgd.yaml
```

## Config Folders

In this section, we will explain the structure of the config folders. Each folder contains a set of config files or subfolders.

### config.yaml

The `sheeprl/configs/config.yaml` is the main configuration, which is loaded by the training scripts. In this config one should find the default configurations:

```yaml
# @package _global_

# Specify here the default training configuration
defaults:
  - _self_
  - algo: default.yaml
  - buffer: default.yaml
  - checkpoint: default.yaml
  - distribution: default.yaml
  - env: default.yaml
  - fabric: default.yaml
  - metric: default.yaml
  - model_manager: default.yaml
  - hydra: default.yaml
  - exp: ???

num_threads: 1
float32_matmul_precision: "high"

# Set it to True to run a single optimization step
dry_run: False

# Reproducibility
seed: 42

# For more information about reproducibility in PyTorch, see https://pytorch.org/docs/stable/notes/randomness.html

# torch.use_deterministic_algorithms() lets you configure PyTorch to use deterministic algorithms
# instead of nondeterministic ones where available,
# and to throw an error if an operation is known to be nondeterministic (and without a deterministic alternative).
torch_use_deterministic_algorithms: False

# Disabling the benchmarking feature with torch.backends.cudnn.benchmark = False
# causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.
# However, if you do not need reproducibility across multiple executions of your application,
# then performance might improve if the benchmarking feature is enabled with torch.backends.cudnn.benchmark = True.
torch_backends_cudnn_benchmark: True

# While disabling CUDA convolution benchmarking (discussed above) ensures that CUDA selects the same algorithm each time an application is run,
# that algorithm itself may be nondeterministic, unless either torch.use_deterministic_algorithms(True)
# or torch.backends.cudnn.deterministic = True is set.
# The latter setting controls only this behavior,
# unlike torch.use_deterministic_algorithms() which will make other PyTorch operations behave deterministically, too.
torch_backends_cudnn_deterministic: False

# From: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
# By design, all cuBLAS API routines from a given toolkit version, generate the same bit-wise results at every run
# when executed on GPUs with the same architecture and the same number of SMs.
# However, bit-wise reproducibility is not guaranteed across toolkit versions
# because the implementation might differ due to some implementation changes.
# This guarantee holds when a single CUDA stream is active only.
# If multiple concurrent streams are active, the library may optimize total performance by picking different internal implementations.
cublas_workspace_config: null  # Possible values are: ":4096:8" or ":16:8"

# Output folders
exp_name: ${algo.name}_${env.id}
run_name: ${now:%Y-%m-%d_%H-%M-%S}_${exp_name}_${seed}
root_dir: ${algo.name}/${env.id}
```

By default we want the user to specify the experiment config, represented by `- exp: ???` in the above example. The three-question-marks symbol tells hydra to expect that an `exp` config is specified at runtime by the user (e.g. `sheeprl.py exp=dreamer_v3`: one can look at every exp configs in `sheeprl/config/exp/` folder).

### Algorithms

In the `algo` folder one can find all the configurations for every algorithm implemented in sheeprl. Those configs contain all the hyperparameters specific to a particular algorithm. Let us have a look at the `dreamer_v3.yaml` config for example:

```yaml
# sheeprl/configs/algo/dreamer_v3.yaml
# Dreamer-V3 XL configuration

defaults:
  - default
  - /optim@world_model.optimizer: adam
  - /optim@actor.optimizer: adam
  - /optim@critic.optimizer: adam
  - _self_

name: dreamer_v3
gamma: 0.996996996996997
lmbda: 0.95
horizon: 15

# Training recipe
replay_ratio: 1
learning_starts: 1024
per_rank_sequence_length: ???

# Encoder and decoder keys
cnn_keys:
  decoder: ${algo.cnn_keys.encoder}
mlp_keys:
  decoder: ${algo.mlp_keys.encoder}

# Model related parameters
layer_norm: True
dense_units: 1024
mlp_layers: 5
dense_act: torch.nn.SiLU
cnn_act: torch.nn.SiLU
unimix: 0.01
hafner_initialization: True
decoupled_rssm: False

# World model
world_model:
  discrete_size: 32
  stochastic_size: 32
  kl_dynamic: 0.5
  kl_representation: 0.1
  kl_free_nats: 1.0
  kl_regularizer: 1.0
  continue_scale_factor: 1.0
  clip_gradients: 1000.0

  # Encoder
  encoder:
    cnn_channels_multiplier: 96
    cnn_act: ${algo.cnn_act}
    dense_act: ${algo.dense_act}
    mlp_layers: ${algo.mlp_layers}
    layer_norm: ${algo.layer_norm}
    dense_units: ${algo.dense_units}

  # Recurrent model
  recurrent_model:
    recurrent_state_size: 4096
    layer_norm: True
    dense_units: ${algo.dense_units}

  # Prior
  transition_model:
    hidden_size: 1024
    dense_act: ${algo.dense_act}
    layer_norm: ${algo.layer_norm}

  # Posterior
  representation_model:
    hidden_size: 1024
    dense_act: ${algo.dense_act}
    layer_norm: ${algo.layer_norm}

  # Decoder
  observation_model:
    cnn_channels_multiplier: ${algo.world_model.encoder.cnn_channels_multiplier}
    cnn_act: ${algo.cnn_act}
    dense_act: ${algo.dense_act}
    mlp_layers: ${algo.mlp_layers}
    layer_norm: ${algo.layer_norm}
    dense_units: ${algo.dense_units}

  # Reward model
  reward_model:
    dense_act: ${algo.dense_act}
    mlp_layers: ${algo.mlp_layers}
    layer_norm: ${algo.layer_norm}
    dense_units: ${algo.dense_units}
    bins: 255

  # Discount model
  discount_model:
    learnable: True
    dense_act: ${algo.dense_act}
    mlp_layers: ${algo.mlp_layers}
    layer_norm: ${algo.layer_norm}
    dense_units: ${algo.dense_units}

  # World model optimizer
  optimizer:
    lr: 1e-4
    eps: 1e-8
    weight_decay: 0

# Actor
actor:
  cls: sheeprl.algos.dreamer_v3.agent.Actor
  ent_coef: 3e-4
  min_std: 0.1
  init_std: 0.0
  objective_mix: 1.0
  dense_act: ${algo.dense_act}
  mlp_layers: ${algo.mlp_layers}
  layer_norm: ${algo.layer_norm}
  dense_units: ${algo.dense_units}
  clip_gradients: 100.0

  # Disttributed percentile model (used to scale the values)
  moments:
    decay: 0.99
    max: 1.0
    percentile:
      low: 0.05
      high: 0.95

  # Actor optimizer
  optimizer:
    lr: 8e-5
    eps: 1e-5
    weight_decay: 0

# Critic
critic:
  dense_act: ${algo.dense_act}
  mlp_layers: ${algo.mlp_layers}
  layer_norm: ${algo.layer_norm}
  dense_units: ${algo.dense_units}
  per_rank_target_network_update_freq: 1
  tau: 0.02
  bins: 255
  clip_gradients: 100.0

  # Critic optimizer
  optimizer:
    lr: 8e-5
    eps: 1e-5
    weight_decay: 0

# Player agent (it interacts with the environment)
player:
  discrete_size: ${algo.world_model.discrete_size}
```

The `defaults` section contains the list of the default configurations to be "imported" by Hydra during the initialization. For more information check the official Hydra documentation about [group defaults](https://hydra.cc/docs/1.1/tutorials/basic/your_first_app/defaults/). The semantic of the following declaration

```yaml
defaults:
  - default
  - /optim@world_model.optimizer: adam
  - /optim@actor.optimizer: adam
  - /optim@critic.optimizer: adam
  - _self_
```

is:

* the content of the `sheeprl/configs/algo/default.yaml` config will be inserted in the current config and whenever a naming collision happens, for example when the same field is defined in both configurations, those will be resolved by keeping the value defined in the current config. This behaviour is specified by letting the `_self_` keyword be the last one in the `defaults` list
* `/optim@world_model.optimizer: adam` (and similar) means that the `adam` config, found in the `sheeprl/configs/optim` folder, will be inserted in this config under the `world_model.optimizer` field, so that one can access it at runtime as `cfg.algo.world_model.optimizer`. As in the previous point, the fields `lr`, `eps`, and `weight_decay` will be overwritten by the one specified in this config

The default configuration for all the algorithms is the following:

```yaml
name: ???
total_steps: ???
per_rank_batch_size: ???

# Encoder and decoder keys
cnn_keys:
  encoder: []
mlp_keys:
  encoder: []
```

> [!WARNING]
>
> Every algorithm config **must** contain the field `name`, the total number of steps `total_steps` and the batch size `per_rank_batch_size`

### Environment

The environment configs can be found under the `sheeprl/configs/env` folders. SheepRL comes with default wrappers for the following environments:

* [Atari](https://gymnasium.farama.org/environments/atari/)
* [Diambra](https://docs.diambra.ai/)
* [Deepmind Control Suite (DMC)](https://github.com/deepmind/dm_control/)
* [Gymnasium](https://www.gymlibrary.dev/)
* [MineRL (v0.4.4)](https://minerl.readthedocs.io/en/v0.4.4/)
* [MineDojo (v0.1.0)](https://docs.minedojo.org/)

In this way, one can easily try out the overall framework with standard RL environments. The `default.yaml` config contains all the environment parameters shared by (possibly) all the environments:

```yaml
id: ???
num_envs: 4
frame_stack: 1
sync_env: False
screen_size: 64
action_repeat: 1
grayscale: False
clip_rewards: False
capture_video: True
frame_stack_dilation: 1
actions_as_observation:
  num_stack: -1
  noop: "You MUST define the NOOP"
  dilation: 1
max_episode_steps: null
reward_as_observation: False
wrapper: ???
```

> [!NOTE]
>
> The actions as observations wrapper is used for adding the last `n` actions to the observations. For more information, check the corresponding [howto file](./actions_as_observation.md).

Every custom environment must then "inherit" from this default config, override the particular parameters, and define the `wrapper` field, which is the one that will be directly instantiated at runtime. The `wrapper` field must define all the specific parameters to be passed to the `_target_` function when the wrapper will be instantiated. Take for example the `atari.yaml` config:

```yaml
defaults:
  - default
  - _self_

# Override from `default` config
action_repeat: 4
id: PongNoFrameskip-v4
max_episode_steps: 27000

# Wrapper to be instantiated
wrapper:
  _target_: gymnasium.wrappers.AtariPreprocessing  # https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.AtariPreprocessing
  env:
    _target_: gymnasium.make
    id: ${env.id}
    render_mode: rgb_array
  noop_max: 30
  terminal_on_life_loss: False
  frame_skip: ${env.action_repeat}
  screen_size: ${env.screen_size}
  grayscale_obs: ${env.grayscale}
  scale_obs: False
  grayscale_newaxis: True
```

> [!WARNING]
>
> Every environment config **must** contain the field `env.id`, which specifies the id of the environment to be instantiated

### Experiment

The `experiment` configs are the main entrypoint for an experiment: it gathers all the different configurations to run a particular experiment in a single configuration file. For example, let us take a look at the `sheeprl/configs/exp/dreamer_v3_100k_ms_pacman.yaml` config:

```yaml
# @package _global_

defaults:
  - dreamer_v3
  - override /env: atari
  - _self_

# Experiment
seed: 5

# Environment
env:
  num_envs: 1
  max_episode_steps: 27000
  id: MsPacmanNoFrameskip-v4

# Checkpoint
checkpoint:
  every: 2000

# Buffer
buffer:
  size: 100000
  checkpoint: True

# Algorithm
algo:
  learning_starts: 1024
  total_steps: 100000

  dense_units: 512
  mlp_layers: 2
  world_model:
    encoder:
      cnn_channels_multiplier: 32
    recurrent_model:
      recurrent_state_size: 512
    transition_model:
      hidden_size: 512
    representation_model:
      hidden_size: 512
```

Given this config, one can easily run an experiment to test the Dreamer-V3 algorithm on the Ms-PacMan environment with the following simple CLI command:

```bash
python sheeprl.py exp=dreamer_v3_100k_ms_pacman
```

> [!WARNING]
>
> The default hyperparameters specified in the configs gathered by the experiment config (in this example the hyperparameters specified by the `sheeprl/configs/exp/dreamer_v3.yaml`, `sheeprl/configs/env/atari.yaml` and all the configs coming with them) will be overwritten by the values in the current config whenever a naming collision happens, for example when the same field is defined in both configurations. Those naming collisions will be resolved by keeping the value defined in the current config. This behaviour is specified by letting the `_self_` keyword be the last one in the `defaults` list.

### Fabric

These configurations control the parameters to be passed to the [Fabric object](https://lightning.ai/docs/fabric/stable/api/generated/lightning.fabric.fabric.Fabric.html#lightning.fabric.fabric.Fabric). With those one can control whether to run the experiments on multiple devices, on which accelerator and with which precision. For more information please have a look at the [Lightning documentation page](https://lightning.ai/docs/fabric/stable/api/fabric_args.html#).

### Hydra

This configuration file manages where and how to create folders or subfolders for experiments. For more information please visit the [hydra documentation](https://hydra.cc/docs/configure_hydra/intro/). Our default Hydra config is the following:

```yaml
run:
  dir: logs/runs/${root_dir}/${run_name}
```

### Metric

The metric config contains all the parameters related to the metrics collected by the algorithm. In SheepRL we make large use of [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/) metrics and in this config we can find both the standard parameters that can be passed to every [Metric](https://torchmetrics.readthedocs.io/en/stable/references/metric.html#torchmetrics.Metric) object and the logging frequency:

```yaml
log_every: 5000

# Metric related parameters. Please have a look at
# https://torchmetrics.readthedocs.io/en/stable/references/metric.html#torchmetrics.Metric
# for more information
sync_on_compute: False
```

### Optimizer

Each optimizer file defines how we initialize the training optimizer with their parameters. For a better understanding of PyTorch optimizers, one should have a look at it at [https://pytorch.org/docs/stable/optim.html](https://pytorch.org/docs/stable/optim.html). An example config is the following:

```yaml
# sheeprl/configs/optim/adam.yaml

_target_: torch.optim.Adam
lr: 2e-4
eps: 1e-04
weight_decay: 0
betas: [0.9, 0.999]
```
