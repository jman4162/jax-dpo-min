import math

import jax
import jax.numpy as jnp
import pytest

from losses import dpo_loss


def _zeros(n=4):
    return jnp.zeros((n,))


def test_policy_equals_ref_gives_log2():
    # When policy == ref everywhere, every logit is 0, so loss = -log sigma(0) = log 2.
    loss, m = dpo_loss(_zeros(), _zeros(), _zeros(), _zeros(), beta=0.1)
    assert jnp.isclose(loss, math.log(2.0), atol=1e-6)
    assert jnp.isclose(m["margin"], 0.0)
    assert jnp.isclose(m["pairwise_accuracy"], 0.0)  # strict >, ties count as wrong


def test_known_closed_form_with_ln2_gaps():
    # Policy log-prob on chosen = ref + ln(2); on rejected = ref - ln(2). With beta=1,
    # margin = 2*ln(2) = ln(4). sigma(ln(4)) = 4/5, so loss = -log(4/5) = log(5/4).
    ref_c, ref_r = jnp.zeros((1,)), jnp.zeros((1,))
    pol_c = ref_c + math.log(2.0)
    pol_r = ref_r - math.log(2.0)
    loss, m = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=1.0)
    assert jnp.isclose(loss, math.log(5.0 / 4.0), atol=1e-6)
    assert jnp.isclose(m["pairwise_accuracy"], 1.0)
    assert jnp.isclose(m["margin"], math.log(4.0))


@pytest.mark.parametrize("beta_small,beta_large", [(0.1, 1.0), (0.5, 2.0)])
def test_beta_monotonicity_when_policy_correct(beta_small, beta_large):
    # Policy favors chosen. Higher beta -> lower loss.
    pol_c = jnp.array([1.0, 1.0])
    pol_r = jnp.array([-1.0, -1.0])
    ref_c = jnp.zeros((2,))
    ref_r = jnp.zeros((2,))
    loss_small, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=beta_small)
    loss_large, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=beta_large)
    assert loss_large < loss_small


def test_gradient_sign_on_policy_logps():
    # Increasing policy_chosen_logps should always decrease loss (grad < 0).
    pol_c = jnp.array([0.0, 0.5])
    pol_r = jnp.array([0.0, -0.5])
    ref_c = jnp.zeros((2,))
    ref_r = jnp.zeros((2,))

    def loss_fn(pc):
        loss, _ = dpo_loss(pc, pol_r, ref_c, ref_r, beta=0.5)
        return loss

    g = jax.grad(loss_fn)(pol_c)
    assert jnp.all(g < 0)

    def loss_fn_rej(pr):
        loss, _ = dpo_loss(pol_c, pr, ref_c, ref_r, beta=0.5)
        return loss

    g_rej = jax.grad(loss_fn_rej)(pol_r)
    assert jnp.all(g_rej > 0)


def test_dpo_no_weight_matches_formula():
    # Without the sigmoid, loss = -mean(beta * margin_logratio).
    pol_c = jnp.array([1.0, 2.0])
    pol_r = jnp.array([-0.5, 0.5])
    ref_c = jnp.array([0.1, 0.2])
    ref_r = jnp.array([0.3, 0.4])
    beta = 0.1
    loss, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=beta, loss_variant="dpo_no_weight")
    expected = -jnp.mean(beta * ((pol_c - ref_c) - (pol_r - ref_r)))
    assert jnp.isclose(loss, expected, atol=1e-6)


def test_dpo_no_ref_ignores_ref_logps():
    # Same chosen/rejected policy logps but wildly different ref values -> same loss.
    pol_c = jnp.array([1.0])
    pol_r = jnp.array([-1.0])
    loss_a, _ = dpo_loss(pol_c, pol_r, jnp.array([0.0]), jnp.array([0.0]), beta=0.5, loss_variant="dpo_no_ref")
    loss_b, _ = dpo_loss(pol_c, pol_r, jnp.array([10.0]), jnp.array([-10.0]), beta=0.5, loss_variant="dpo_no_ref")
    assert jnp.isclose(loss_a, loss_b, atol=1e-6)


def test_sft_chosen_variant():
    # SFT-on-chosen baseline: loss is negative mean of policy_chosen_logps.
    pol_c = jnp.array([-2.0, -4.0])
    loss, _ = dpo_loss(pol_c, jnp.zeros_like(pol_c), jnp.zeros_like(pol_c), jnp.zeros_like(pol_c),
                      beta=1.0, loss_variant="sft_chosen")
    assert jnp.isclose(loss, 3.0, atol=1e-6)
