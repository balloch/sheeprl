from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Sequence
import os

import imageio
import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor, nn

from sheeprl.utils.env import make_env
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE
from sheeprl.utils.utils import unwrap_fabric

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo

    from sheeprl.algos.offline_dreamer.agent import PlayerODV3

# constants
CONCEPT_DICT = {
    0: 'white_yellow_mug',
    1: 'butter',
    2: 'wine_bottle',
    3: 'yellow_book',
    4: 'ketchup',
    5: 'tomato_sauce',
    6: 'orange_juice',
    7: 'porcelain_mug',
    8: 'chefmate_8_frypan',
    9: 'cream_cheese',
    10: 'plate',
    11: 'chocolate_pudding',
    12: 'red_coffee_mug',
    13: 'moka_pot',
    14: 'basket',
    15: 'milk',
    16: 'white_bowl',
    17: 'wooden_tray',
    18: 'akita_black_bowl',
    19: 'alphabet_soup',
    20: 'black_book',
    21: 'new_salad_dressing',
}
AGGREGATOR_KEYS = {
    "Rewards/rew_avg",
    "Rewards/ep_rew_max",
    "Rewards/ep_rew_mean",
    "Game/ep_len_avg",
    "Loss/world_model_loss",
    "Loss/value_loss",
    "Loss/policy_loss",
    "Loss/observation_loss",
    "Loss/reward_loss",
    "Loss/state_loss",
    "Loss/continue_loss",
    "Loss/concept_loss",
    "Loss/per_concept_loss",
    "Loss/orthognality_loss",
    "State/kl",
    "State/post_entropy",
    "State/prior_entropy",
    "Grads/world_model",
    "Grads/actor",
    "Grads/critic",
    "Rewards/rew_avg_ep",
    "Rewards/rew_max_ep",
    "Rewards/reach_avg_ep",
    "Rewards/reach_max_ep",
    "Rewards/grasp_avg_ep",
    "Rewards/grasp_max_ep",
    "Rewards/lift_avg_ep",
    "Rewards/lift_max_ep",
    "Rewards/hover_avg_ep",
    "Rewards/hover_max_ep",
    # "trainer/wm_train_iter",
    # "trainer/policy_train_iter",
    # "trainer/env_steps",
    "Val/world_model_loss",
    "Val/observation_loss",
    "Val/observation_error",
    "Val/reward_loss",
    "Val/state_loss",
    "Val/continue_loss",
    "Val/concept_loss",
    "Val/per_concept_loss",
    "Val/concept_precision",
    "Val/concept_recall",
    "Val/concept_f1_score",
    "Val/concept_accuracy",
    "Val/orthognality_loss",
}
MODELS_TO_REGISTER = {"world_model", "actor", "critic", "target_critic", "moments"}

### TODO: We need these functions but they do not work as written because of implementation complexity
# def reset_model(model, module_list=None):
#     for module in model.modules() and module in module_list:
#         if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm)):
#             module.reset_parameters()

# def freeze_weights(model, module_list=None):
#     for module in model.modules() and module in module_list:
#         if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm)):
#             module.requires_grad = False
#         if isinstance(module, (nn.BatchNorm2d,nn.LayerNorm)):
#             module.eval()
#             module.track_running_stats = False

class Moments(nn.Module):
    def __init__(
        self,
        decay: float = 0.99,
        max_: float = 1e8,
        percentile_low: float = 0.05,
        percentile_high: float = 0.95,
    ) -> None:
        super().__init__()
        self._decay = decay
        self._max = torch.tensor(max_)
        self._percentile_low = percentile_low
        self._percentile_high = percentile_high
        self.register_buffer("low", torch.zeros((), dtype=torch.float32))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32))

    def forward(self, x: Tensor, fabric: Fabric) -> Any:
        gathered_x = fabric.all_gather(x).float().detach()
        low = torch.quantile(gathered_x, self._percentile_low)
        high = torch.quantile(gathered_x, self._percentile_high)
        self.low = self._decay * self.low + (1 - self._decay) * low
        self.high = self._decay * self.high + (1 - self._decay) * high
        invscale = torch.max(1 / self._max, self.high - self.low)
        return self.low.detach(), invscale.detach()


def compute_lambda_values(
    rewards: Tensor,
    values: Tensor,
    continues: Tensor,
    lmbda: float = 0.95,
):
    vals = [values[-1:]]
    interm = rewards + continues * values * (1 - lmbda)
    for t in reversed(range(len(continues))):
        vals.append(interm[t] + continues[t] * lmbda * vals[-1])
    ret = torch.cat(list(reversed(vals))[:-1])
    return ret


def prepare_obs(
    fabric: Fabric, obs: Dict[str, np.ndarray], *, cnn_keys: Sequence[str] = [], num_envs: int = 1, **kwargs
) -> Dict[str, Tensor]:
    torch_obs = {}
    for k, v in obs.items():
        torch_obs[k] = torch.from_numpy(v.copy()).to(fabric.device).float()
        if k in cnn_keys:
            torch_obs[k] = torch_obs[k].view(1, num_envs, -1, *v.shape[-2:]) / 255 - 0.5
        else:
            torch_obs[k] = torch_obs[k].view(1, num_envs, -1)

    return torch_obs


@torch.no_grad()
def test(
    player: "PlayerODV3",
    fabric: Fabric,
    cfg: Dict[str, Any],
    log_dir: str,
    test_name: str = "",
    greedy: bool = True,
):
    """Test the model on the environment with the frozen model.

    Args:
        player (PlayerODV3): the agent which contains all the models needed to play.
        fabric (Fabric): the fabric instance.
        cfg (DictConfig): the hyper-parameters.
        log_dir (str): the logging directory.
        test_name (str): the name of the test.
            Default to "".
        greedy (bool): whether or not to sample the actions.
            Default to True.
    """
    env: gym.Env = make_env(cfg, cfg.seed, 0, log_dir, "test" + (f"_{test_name}" if test_name != "" else ""))()
    done = False
    cumulative_rew = 0
    obs = env.reset(seed=cfg.seed)[0]
    player.num_envs = 1
    player.init_states()
    while not done:
        # Act greedly through the environment
        torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder)
        real_actions = player.get_actions(
            torch_obs, greedy, {k: v for k, v in torch_obs.items() if k.startswith("mask")}
        )
        if player.actor.is_continuous:
            real_actions = torch.stack(real_actions, -1).cpu().numpy()
        else:
            real_actions = torch.stack([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()

        # Single environment step
        obs, reward, done, truncated, _ = env.step(real_actions.reshape(env.action_space.shape))
        done = done or truncated or cfg.dry_run
        cumulative_rew += reward
    fabric.print("Test - Reward:", cumulative_rew)
    if cfg.metric.log_level > 0 and len(fabric.loggers) > 0:
        fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()


# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L929
def init_weights(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L957
def uniform_init_weights(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence["ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    import mlflow  # noqa

    from sheeprl.algos.offline_dreamer.agent import build_agent

    # Create the models
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        env.action_space.shape
        if is_continuous
        else (env.action_space.nvec.tolist() if is_multidiscrete else [env.action_space.n])
    )
    world_model, actor, critic, target_critic = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        env.observation_space,
        state["world_model"],
        state["actor"],
        state["critic"],
        state["target_critic"],
    )
    moments = Moments(
        fabric,
        cfg.algo.actor.moments.decay,
        cfg.algo.actor.moments.max,
        cfg.algo.actor.moments.percentile.low,
        cfg.algo.actor.moments.percentile.high,
    )
    moments.load_state_dict(state["moments"])

    # Log the model, create a new run if `cfg.run_id` is None.
    model_info = {}
    with mlflow.start_run(run_id=cfg.run.id, experiment_id=cfg.experiment.id, run_name=cfg.run.name, nested=True) as _:
        model_info["world_model"] = mlflow.pytorch.log_model(unwrap_fabric(world_model), artifact_path="world_model")
        model_info["actor"] = mlflow.pytorch.log_model(unwrap_fabric(actor), artifact_path="actor")
        model_info["critic"] = mlflow.pytorch.log_model(unwrap_fabric(critic), artifact_path="critic")
        model_info["target_critic"] = mlflow.pytorch.log_model(target_critic, artifact_path="target_critic")
        model_info["moments"] = mlflow.pytorch.log_model(moments, artifact_path="moments")
        mlflow.log_dict(cfg.to_log, "config.json")
    return model_info


def calculate_mean_emb(pos_emb_array, TP_hits):
    pos_emb_array = pos_emb_array.reshape(pos_emb_array.shape[:-1] + (TP_hits.shape[-1], -1))
    pos_emb_array = pos_emb_array.reshape(-1, pos_emb_array.shape[-2], pos_emb_array.shape[-1])
    pos_emb_array = np.swapaxes(pos_emb_array, 0, 1)
    TP_hits = TP_hits.reshape(-1, TP_hits.shape[-1])
    TP_hits = np.swapaxes(TP_hits, 0, 1)
    mean_emb_dict = {}
    for i, (pos_emb, TP_concept) in enumerate(zip(pos_emb_array, TP_hits)):
        pos_emb = pos_emb[TP_concept, :]
        if pos_emb.shape[0] > 0:
            mean_emb_dict[i] = np.mean(pos_emb, axis=0).reshape(1, -1)
    return mean_emb_dict


def compare_concepts(embedding_path1, tp_path1, embedding_path2, tp_path2):
    emb_arr1 = np.load(embedding_path1)
    tp_arr1 = np.load(tp_path1).astype(int) > 0
    emb_arr2 = np.load(embedding_path2)
    tp_arr2 = np.load(tp_path2).astype(int) > 0
    mean_embed_dict1 = calculate_mean_emb(emb_arr1, tp_arr1)
    mean_embed_dict2 = calculate_mean_emb(emb_arr2, tp_arr2)
    for key in mean_embed_dict1.keys():
        if key in mean_embed_dict2.keys():
            print(mean_embed_dict1[key].shape)
            cos_sim = cosine_similarity(mean_embed_dict1[key], mean_embed_dict2[key])
            print('concept:', CONCEPT_DICT[key], 'cosine similarity:', cos_sim)

            
def render_vid(
    fabric,
    orig_obs,
    reconstructed_obs,
    cfg,
    orig_obs_prepared=False,
    recon_obs_prepared=False,
    obs_key='obs',
    log_dir=None,
    logger=None,
    train_iter=None,
    ):
    """
    Render a video of the original and reconstructed observations.

    Args:
        fabric (Fabric): the fabric instance.
        orig_obs (Tensor): the original observations.
        reconstructed_obs (Tensor): the reconstructed observations.
        cfg (DictConfig): the hyper-parameters.
        obs_key (str): the observation key.
            Default to "obs".
        log_dir (str): the logging directory.
            Default to None.
        logger (Logger): the logger.
            Default to None.
        train_iter (int): the training iteration.
            Default to None.
    """
    assert logger or log_dir, "Either logger or log_dir must be provided"

    obs_video = orig_obs.permute(0,1,3,4,2)[:,cfg.seed,...]
    recon_video = reconstructed_obs.permute(0,1,3,4,2)[:,cfg.seed,...]
    if not cfg.algo.world_model.observation_model.final_sigmoid:
        obs_video = obs_video + 0.5
        recon_video = recon_video + 0.5
        recon_video[recon_video<0] = 0
        recon_video[recon_video>1] = 1
    obs_video = (obs_video * 255).cpu().numpy().astype(np.uint8)
    recon_video = (recon_video * 255).cpu().numpy().astype(np.uint8)

    os.makedirs(log_dir, exist_ok=True)
    # os.chmod(log_dir, 0o777)

    filename, increment = increment_filename(log_dir, f'recon_{obs_key}_vid.mp4')

    if cfg.validate.save_to_file:
        with open(os.path.join(log_dir, filename), 'wb') as f:
            imageio.v3.imwrite(
                uri=f,
                image=recon_video,
                plugin='FFMPEG',
                extension='.mp4',
                )
        with open(os.path.join(log_dir, f'orig_{obs_key}_vid'+str(increment)+'.mp4'), 'wb') as of:
            imageio.v3.imwrite(
                uri=of,
                image=obs_video,
                plugin='FFMPEG',
                extension='.mp4',
                )


def increment_filename(directory, filename):
    if filename[0] == '/':
        filename = filename[1:]
    name, ext = os.path.splitext(filename)
    version = 0
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        # Create a new filename by appending the version number
        version += 1
        new_filename = f"{name}_{version}{ext}"
    if filename == new_filename:
        version = ''
    return new_filename, version

    # if "wandb" in cfg.metric.logger._target_.lower() and qualitative_log is False:
    #     wandb.log({"data_video": wandb.Video(
    #         np.transpose((data['agentview_rgb'][:,0,...].cpu().numpy()* 255).astype(np.uint8),(0, 2, 3, 1)),
    #         fps=10),
    #         # np.transpose(video, (0, 2, 3, 1))
    #         # fps=10
    #         # imageio.mimsave(output_path, video, fps=fps)
    #         #    "predicted_video": wandb.Video(reconstructed_obs['agentview_rgb'][:,0,...].cpu().detach().numpy(), duration=5)
    #         })
    #     qualitative_log = True
    # torch.nn.functional.sigmoid(reconstructed_obs['agentview_rgb'])
