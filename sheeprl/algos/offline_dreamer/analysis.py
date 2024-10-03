from functools import partial

import gym
import numpy as np
import torch
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from lightning import Fabric
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import v2

from sheeprl.algos.offline_dreamer.agent import CBWM, WorldModel, build_agent
from sheeprl.algos.offline_dreamer.offline_dreamer import (
    CombinedDictDataset,
    TransformedDictDataset,
    dynamic_learning,
    get_datasets_from_benchmark,
    validate_wm,
)
from sheeprl.envs.wrappers import RestartOnException
from sheeprl.utils.env import make_env

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
            mean_emb_dict[i] = np.mean(pos_emb, axis=0)
    return mean_emb_dict


def main(cfg, model1, model2, env_path, mode):
    # Setup Fabric
    strategy = cfg.fabric.get("strategy", "auto")
    fabric: Fabric = hydra.utils.instantiate(cfg.fabric, strategy=strategy, _convert_="all")

    if mode == 'collect':
        # create dataset
        compiled_dynamic_learning = torch.compile(dynamic_learning, **cfg.algo.compile_dynamic_learning)
        libero_folder = get_libero_path("datasets")
        obs_modality = {'rgb': ['agentview_rgb']}
        task_order = 0
        benchmark_name = "libero_90" 
        benchmark = get_benchmark(benchmark_name)(task_order)
        datasets, descriptions, task_concepts = get_datasets_from_benchmark(
            benchmark=benchmark,
            libero_folder=libero_folder,
            seq_len=cfg.algo.per_rank_sequence_length,
            obs_modality=obs_modality
            )
        if cfg.algo.supervised_concepts:
            temp_datas = []
            for dataset, task_concept in zip(datasets, task_concepts):
                temp_datas.append(CombinedDictDataset(dataset, task_concept))
            datasets = temp_datas
        concat_dataset = ConcatDataset(datasets)
        generator1 = torch.Generator().manual_seed(cfg.seed)
        train_split, val_split = torch.utils.data.random_split(
            dataset=concat_dataset,
            lengths=[cfg.algo.offline_train_split, 1-cfg.algo.offline_train_split],
            generator=generator1
        )
        val_transforms_dict = {
            'agentview_rgb': v2.Compose([
                # v2.ToImage(),
                v2.Resize((cfg.env.screen_size, cfg.env.screen_size)),
                # v2.Pad(4,padding_mode=cfg.val_transforms.Pad.pad_type),
                # v2.RandomCrop(cfg.env.screen_size),
                # v2.ToTensor(),
        ])}
        val_dataset = TransformedDictDataset(
            dataset=val_split,
            transform_dict=val_transforms_dict
            )
        dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.algo.per_rank_batch_size, # replay_ratio=0.5 This is hacky, but Ratio is confusing
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            )

        # create environment
        rank = fabric.global_rank
        vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
        envs = vectorized_env(
            [
                partial(
                    RestartOnException,
                    make_env(
                        cfg,
                        cfg.seed + rank * cfg.env.num_envs + i,
                        rank * cfg.env.num_envs,
                        None,
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

        # create models
        state1 = fabric.load(pathlib.Path(model1))
        state2 = fabric.load(pathlib.Path(model2))
        world_model1, actor1, critic1, target_critic1, player1 = build_agent(
            fabric,
            actions_dim,
            is_continuous,
            cfg,
            observation_space,
            state1["world_model"],
            state1["actor"],
            state1["critic"],
            state1["target_critic"],
        )
        world_model2, actor2, critic2, target_critic2, player2 = build_agent(
            fabric,
            actions_dim,
            is_continuous,
            cfg,
            observation_space,
            state2["world_model"],
            state2["actor"],
            state2["critic"],
            state2["target_critic"],
        )
        validate_wm(
            fabric, # just need for fabric.device
            world_model1,
            dataloader,
            cfg,
            compiled_dynamic_learning,
            save_embeddings=True,
        )
        validate_wm(
            fabric,
            world_model2,
            dataloader,
            cfg,
            compiled_dynamic_learning,
            save_embeddings=True,
        )
    elif mode == 'compare':
        mean_embed_dict1 = calculate_mean_emb(embedding_path1, tp_path1)
        mean_embed_dict2 = calculate_mean_emb(embedding_path2, tp_path2)
        for key in mean_embed_dict1.keys():
            if key in mean_embed_dict2.keys():
                cos_sim = cosine_similarity(mean_embed_dict1[key], mean_embed_dict2[key])
                print('concept:', CONCEPT_DICT[key], 'cosine similarity:', cos_sim)



if __name__ == '__main__':
    main()