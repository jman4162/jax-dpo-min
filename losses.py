"""DPO loss — paper Eq. 7 (Rafailov et al., 2023).

For a pair (y_w, y_l) with prompt x:
    L_DPO = -log sigma( beta * ( (log pi_theta(y_w|x) - log pi_ref(y_w|x))
                                - (log pi_theta(y_l|x) - log pi_ref(y_l|x)) ) )

The sigmoid is the "weighting term" the paper warns against removing: it
down-weights pairs the policy already ranks correctly, preventing runaway
log-ratios. `dpo_no_weight` removes it for an ablation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def dpo_loss(
    policy_chosen_logps: jnp.ndarray,
    policy_rejected_logps: jnp.ndarray,
    ref_chosen_logps: jnp.ndarray,
    ref_rejected_logps: jnp.ndarray,
    beta: float,
    *,
    loss_variant: str = "dpo",
) -> tuple[jnp.ndarray, dict]:
    if loss_variant == "dpo":
        chosen_logratio = policy_chosen_logps - ref_chosen_logps
        rejected_logratio = policy_rejected_logps - ref_rejected_logps
    elif loss_variant == "dpo_no_ref":
        chosen_logratio = policy_chosen_logps
        rejected_logratio = policy_rejected_logps
    elif loss_variant == "dpo_no_weight":
        chosen_logratio = policy_chosen_logps - ref_chosen_logps
        rejected_logratio = policy_rejected_logps - ref_rejected_logps
    elif loss_variant == "sft_chosen":
        loss = -policy_chosen_logps.mean()
        metrics = {
            "chosen_reward": jnp.zeros(()),
            "rejected_reward": jnp.zeros(()),
            "margin": jnp.zeros(()),
            "pairwise_accuracy": jnp.zeros(()),
            "logits": jnp.zeros(()),
        }
        return loss, metrics
    else:
        raise ValueError(f"unknown loss_variant: {loss_variant}")

    chosen_reward = beta * chosen_logratio
    rejected_reward = beta * rejected_logratio
    margin = chosen_reward - rejected_reward

    loss = -margin.mean() if loss_variant == "dpo_no_weight" else -jax.nn.log_sigmoid(margin).mean()

    metrics = {
        "chosen_reward": chosen_reward.mean(),
        "rejected_reward": rejected_reward.mean(),
        "margin": margin.mean(),
        "pairwise_accuracy": (chosen_reward > rejected_reward).astype(jnp.float32).mean(),
        "logits": margin.mean(),
    }
    return loss, metrics
