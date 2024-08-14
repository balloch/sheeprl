from typing import Dict, Optional, Tuple

import torch
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor
from torch.distributions import Distribution, Independent, OneHotCategoricalStraightThrough
from torch.distributions.kl import kl_divergence


def get_concept_index(model, c):
    if c==0:
        start=0
    else:
        start=sum(model.concept_bins[:c])
    end= sum(model.concept_bins[:c+1])

    return start, end


def get_concept_loss(model, predicted_concepts, target_concepts, isList=False):
    ## TODO im not even sure this is right...it seems like from the paper that the label predictor is supposed to take 2x embeddings as input
    ## and then predict n_concept labels. But here we predicting n_concept labels?
    concept_loss = 0
    loss_bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    # loss_ce = torch.nn.CrossEntropyLoss()
    target_concepts = target_concepts.repeat_interleave(repeats=model.concept_bins[0],dim=-1)  # To supervise the doubled concept predictions
    # import pdb; pdb.set_trace()
    losses = loss_bce(predicted_concepts, target_concepts)
    loss_per_concept = losses.view(
        losses.shape[0],
        losses.shape[1],
        model.n_concepts,
        model.concept_bins[0]).mean(dim=[0,1,-1])
    concept_loss = loss_per_concept.mean()
    # concept_loss_lst=[]
    # for c in range(model.n_concepts):  # Todo this is just counting by twos. inefficient?
    #     start,end = get_concept_index(model,c)
    #     c_predicted_concepts=predicted_concepts[:,start:end]
    #     if(not isList):
    #         c_real_concepts=target_concepts[:,start:end]
    #     else:
    #         c_real_concepts=target_concepts[c]
    #     c_real_concepts=c_real_concepts.to(c_predicted_concepts.device) # todo their might be a better way to do this
    #     c_concept_loss = loss_ce(c_predicted_concepts, c_real_concepts)  # Mean
    #     concept_loss+=c_concept_loss # Sum??
    #     concept_loss_lst.append(c_concept_loss)
    ### TODO Why mixing mean and sum like this?
    # return concept_loss, concept_loss_lst

    return concept_loss, loss_per_concept


def OrthogonalProjectionLoss(embed1, embed2):
    #  features are normalized
    embed1 = torch.nn.functional.normalize(embed1, dim=1)
    embed2 = torch.nn.functional.normalize(embed2, dim=1)

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    output = torch.abs(cos(embed1, embed2))
    return output.mean()


def reconstruction_loss(
    po: Dict[str, Distribution],
    observations: Tensor,
    pr: Distribution,
    rewards: Tensor,
    priors_logits: Tensor,
    posteriors_logits: Tensor,
    world_model: _FabricModule,
    cem_data: None | Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    use_cbm: bool,
    kl_dynamic: float = 0.5,
    kl_representation: float = 0.1,
    kl_free_nats: float = 1.0,
    kl_regularizer: float = 1.0,
    pc: Optional[Distribution] = None,
    continue_targets: Optional[Tensor] = None,
    continue_scale_factor: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the reconstruction loss as described in Eq. 5 in
    [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104).

    Args:
        po (Dict[str, Distribution]): the distribution returned by the observation_model (decoder).
        observations (Tensor): the observations provided by the environment.
        pr (Distribution): the reward distribution returned by the reward_model.
        rewards (Tensor): the rewards obtained by the agent during the "Environment interaction" phase.
        priors_logits (Tensor): the logits of the prior.
        posteriors_logits (Tensor): the logits of the posterior.
        kl_dynamic (float): the kl-balancing dynamic loss regularizer.
            Defaults to 0.5.
        kl_balancing_alpha (float): the kl-balancing representation loss regularizer.
            Defaults to 0.1.
        kl_free_nats (float): lower bound of the KL divergence.
            Default to 1.0.
        kl_regularizer (float): scale factor of the KL divergence.
            Default to 1.0.
        pc (Bernoulli, optional): the predicted Bernoulli distribution of the terminal steps.
            0s for the entries that are relative to a terminal step, 1s otherwise.
            Default to None.
        continue_targets (Tensor, optional): the targets for the discount predictor. Those are normally computed
            as `(1 - data["dones"]) * args.gamma`.
            Default to None.
        continue_scale_factor (float): the scale factor for the continue loss.
            Default to 10.

    Returns:
        observation_loss (Tensor): the value of the observation loss.
        KL divergence (Tensor): the KL divergence between the posterior and the prior.
        reward_loss (Tensor): the value of the reward loss.
        state_loss (Tensor): the value of the state loss.
        continue_loss (Tensor): the value of the continue loss (0 if it is not computed).
        reconstruction_loss (Tensor): the value of the overall reconstruction loss.
    """
    rewards.device
    observation_loss = -sum([po[k].log_prob(observations[k]) for k in po.keys()])
    reward_loss = -pr.log_prob(rewards)
    # KL balancing
    dyn_loss = kl = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1),
        Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1),
    )
    free_nats = torch.full_like(dyn_loss, kl_free_nats)
    dyn_loss = kl_dynamic * torch.maximum(dyn_loss, free_nats)
    repr_loss = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 1),
        Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1),
    )
    repr_loss = kl_representation * torch.maximum(repr_loss, free_nats)
    kl_loss = dyn_loss + repr_loss
    if pc is not None and continue_targets is not None:
        continue_loss = continue_scale_factor * -pc.log_prob(continue_targets)
    else:
        continue_loss = torch.zeros_like(reward_loss)

    loss_dict = {
        'kl':kl.mean(),
        'kl_loss':kl_loss.mean(),
        'reward_loss':reward_loss.mean(),
        'observation_loss':observation_loss.mean(),
        'continue_loss':continue_loss.mean(),
    }
    if use_cbm is False:
        reconstruction_loss = (kl_regularizer * kl_loss + observation_loss + reward_loss + continue_loss).mean()
    else:
        #TODO replace with actual concepts
        pred_concepts, target_concepts, real_concept_latent, real_non_concept_latent, rand_concept_latent, rand_non_concept_latent = cem_data
        if target_concepts is None:
            target_concepts = (torch.rand(pred_concepts.size()) > 0.5) * 1  # TODO replace with actual concepts
            print("Randomly generated target concepts")
        target_concepts = target_concepts.float()
        pred_concepts = pred_concepts.float()
        concept_loss, loss_per_concept = get_concept_loss(world_model.cem, pred_concepts, target_concepts)
        loss_dict['concept_loss'] = concept_loss.mean()
        loss_dict['loss_per_concept'] = loss_per_concept
        orthognality_loss = []
        for c in range(world_model.cem.n_concepts):  #TODO why sum?
            orthognality_loss.append(OrthogonalProjectionLoss(
                real_concept_latent[:, :, c*world_model.cem.emb_size: (c*world_model.cem.emb_size) + world_model.cem.emb_size],
                real_non_concept_latent))
            orthognality_loss.append(OrthogonalProjectionLoss(
                rand_concept_latent[:, :, c*world_model.cem.emb_size: (c*world_model.cem.emb_size) + world_model.cem.emb_size],
                rand_non_concept_latent))
        loss_dict['orthognality_loss'] = torch.stack(orthognality_loss).mean()
        cbm_loss = concept_loss + loss_dict['orthognality_loss']
        loss_dict['cbm_loss'] = cbm_loss.mean()

        reconstruction_loss = (kl_regularizer * kl_loss + observation_loss + reward_loss + continue_loss + cbm_loss).mean()

    return (
        reconstruction_loss,
        loss_dict
    )
