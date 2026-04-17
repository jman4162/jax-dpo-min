"""Load a HuggingFace Flax causal LM and score sequence log-probs.

Kept separate from `train.py` so `eval.py` and notebooks can reuse the scorer
without pulling in the training loop.
"""
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

# Map from user-facing model name to the Linen leaf modules that should
# receive LoRA adapters. Extend here if you add a new model family.
LORA_TARGETS = {
    "gpt2": ["c_attn"],
    "gpt2-medium": ["c_attn"],
    "distilgpt2": ["c_attn"],
    "google/gemma-2b-it": ["q_proj", "v_proj"],
    "google/gemma-2b": ["q_proj", "v_proj"],
}


_DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}


def load_model_and_tokenizer(name: str, dtype: str = "float32"):
    """Load a HF Flax causal LM. Returns (model, params, tokenizer).

    `dtype="bfloat16"` casts both compute and params to bf16 — required for
    Gemma-2B on a TPU v2 core (8 GB HBM). Optimizer moments will also be bf16,
    which is fine for a tutorial-scale run; production would keep fp32 master
    params.
    """
    from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {list(_DTYPES)}, got {dtype!r}")
    jdtype = _DTYPES[dtype]

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FlaxAutoModelForCausalLM.from_pretrained(name, dtype=jdtype)
    params = jax.tree_util.tree_map(lambda x: x.astype(jdtype), model.params)
    return model, params, tokenizer


def compute_logps(
    apply_fn: Callable,
    params,
    input_ids: jnp.ndarray,       # (B, T) int32
    attention_mask: jnp.ndarray,  # (B, T) int32
    completion_mask: jnp.ndarray, # (B, T) int32
) -> jnp.ndarray:
    """Sum log p(next_token | prefix) over completion tokens only.

    Uses a causal shift: logits at position t predict token at t+1, so we
    compare logits[:, :-1] against input_ids[:, 1:] and mask with
    completion_mask[:, 1:]. This keeps prompt tokens (and pad) out of the sum.
    """
    logits = apply_fn(input_ids=input_ids, attention_mask=attention_mask, params=params).logits
    # (B, T-1, V) vs (B, T-1)
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = completion_mask[:, 1:].astype(jnp.float32)

    logprobs = jax.nn.log_softmax(shift_logits, axis=-1)
    token_logp = jnp.take_along_axis(logprobs, shift_labels[..., None], axis=-1).squeeze(-1)
    return (token_logp * shift_mask).sum(axis=-1)
