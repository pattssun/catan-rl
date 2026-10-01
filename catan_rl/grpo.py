"""The GRPO objective, independent of model loading and generation."""

import torch


def group_advantages(rewards, epsilon=1e-4):
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("Rewards must be [prompts, group_size >= 2]")
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    return centered / (rewards.std(dim=1, keepdim=True, unbiased=False) + epsilon)


def grpo_loss(log_probs, old_log_probs, reference_log_probs, advantages, mask,
              clip=0.2, beta=0.02):
    if log_probs.shape != old_log_probs.shape or log_probs.shape != mask.shape:
        raise ValueError("Token log probabilities and completion mask must match")
    if not torch.all(mask.sum(dim=1) > 0):
        raise ValueError("Every completion must have at least one scored token")
    old = old_log_probs.detach()
    reference = reference_log_probs.detach()
    advantage = advantages.detach().reshape(-1, 1)
    ratio = (log_probs - old).exp()
    clipped = ratio.clamp(1 - clip, 1 + clip)
    objective = torch.minimum(ratio * advantage, clipped * advantage)
    log_ref_ratio = reference - log_probs
    kl = log_ref_ratio.exp() - log_ref_ratio - 1
    tokens = mask.sum(dim=1)
    loss = (((-objective + beta * kl) * mask).sum(dim=1) / tokens).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite GRPO loss")
    with torch.no_grad():
        metrics = {"kl": ((kl * mask).sum(dim=1) / tokens).mean().item(),
                   "clip_fraction": ((((ratio - 1).abs() > clip) * mask).sum()
                                     / mask.sum()).item()}
    return loss, metrics
