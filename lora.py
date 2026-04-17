"""Minimal LoRA as a parameter delta.

The frozen base model + a LoRA delta gives us the policy; the frozen base
model alone gives us the reference. One param tree, two forward passes. This
is mathematically equivalent to the adapter-style LoRA but sits cleanly on
top of `transformers`' Flax modules without subclassing them.

For each targeted `kernel` weight W of shape (in, out), we store:
    A: (in, rank)    random init (scaled)
    B: (rank, out)   zero init
and at forward time compute W + (alpha / rank) * A @ B.

B=0 init means the untrained policy equals the reference, which is the right
starting point for DPO.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict


def _as_plain_dict(params) -> dict:
    # `transformers` Flax modules hand back a FrozenDict in older versions and
    # a plain dict in newer ones. Normalize.
    try:
        return unfreeze(params)
    except TypeError:
        return dict(params)


def init_lora(
    params,
    target_modules: list[str],
    rank: int,
    alpha: float,
    seed: int = 0,
) -> dict:
    """Return a dict {path_key: {"A": ndarray, "B": ndarray}} for every
    kernel whose parent module name matches `target_modules`.
    """
    flat = flatten_dict(_as_plain_dict(params))
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(rank)
    lora = {}
    for path, leaf in flat.items():
        if path[-1] != "kernel":
            continue
        parent = path[-2] if len(path) >= 2 else ""
        if parent not in target_modules:
            continue
        in_dim, out_dim = leaf.shape
        A = jnp.asarray(rng.standard_normal((in_dim, rank)) * scale, dtype=leaf.dtype)
        B = jnp.zeros((rank, out_dim), dtype=leaf.dtype)
        lora["/".join(path)] = {"A": A, "B": B}
    if not lora:
        raise ValueError(f"No LoRA targets matched any of {target_modules}")
    return lora


def merge(base_params, lora_params: dict, alpha: float, rank: int):
    """Return base_params with LoRA deltas added in place of each targeted kernel."""
    scale = alpha / rank
    flat = flatten_dict(_as_plain_dict(base_params))
    for key, ab in lora_params.items():
        path = tuple(key.split("/"))
        delta = scale * (ab["A"] @ ab["B"])
        flat[path] = flat[path] + delta
    return freeze(unflatten_dict(flat))


def count_params(tree: Any) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(tree)))
