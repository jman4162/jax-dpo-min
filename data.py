"""Tokenize preference pairs into fixed-shape tensors for a JAX train step.

For each example we produce, for both the chosen and rejected completion:
    input_ids       : [prompt_tokens, response_tokens, pad...]  shape (T,)
    attention_mask  : 1 on real tokens, 0 on pad                 shape (T,)
    completion_mask : 1 on response tokens *only*, else 0        shape (T,)

The completion mask is the single most error-prone piece of a DPO
implementation. Log-probs must be scored on the response tokens only, not on
the prompt. Getting this wrong still produces a decreasing loss — it just
trains the wrong thing.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _encode_one(tokenizer, prompt: str, response: str, max_length: int):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if eos is not None:
        response_ids = response_ids + [eos]

    full = prompt_ids + response_ids
    if len(full) > max_length:
        # Drop prompt tokens from the left first (completion is more valuable).
        # If the response alone exceeds max_length, also truncate the response from the right.
        if len(response_ids) >= max_length:
            prompt_ids = []
            response_ids = response_ids[:max_length]
        else:
            overflow = len(full) - max_length
            prompt_ids = prompt_ids[overflow:]
        full = prompt_ids + response_ids

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    pad_len = max_length - len(full)

    input_ids = np.array(full + [pad_id] * pad_len, dtype=np.int32)
    attention_mask = np.zeros(max_length, dtype=np.int32)
    attention_mask[: len(full)] = 1
    completion_mask = np.zeros(max_length, dtype=np.int32)
    completion_mask[len(prompt_ids) : len(prompt_ids) + len(response_ids)] = 1
    return input_ids, attention_mask, completion_mask


def build_example(tokenizer, prompt: str, chosen: str, rejected: str, max_length: int):
    c_ids, c_amask, c_cmask = _encode_one(tokenizer, prompt, chosen, max_length)
    r_ids, r_amask, r_cmask = _encode_one(tokenizer, prompt, rejected, max_length)
    return {
        "chosen_input_ids": c_ids,
        "chosen_attention_mask": c_amask,
        "chosen_completion_mask": c_cmask,
        "rejected_input_ids": r_ids,
        "rejected_attention_mask": r_amask,
        "rejected_completion_mask": r_cmask,
    }


def build_batch(examples: Iterable[dict], tokenizer, max_length: int) -> dict:
    rows = [
        build_example(tokenizer, ex["prompt"], ex["chosen"], ex["rejected"], max_length)
        for ex in examples
    ]
    return {k: np.stack([r[k] for r in rows], axis=0) for k in rows[0]}


def iter_batches(dataset, tokenizer, batch_size: int, max_length: int, *, shuffle_seed: int | None = None):
    """Yield dict batches forever, reshuffling each epoch."""
    n = len(dataset)
    rng = np.random.default_rng(shuffle_seed)
    while True:
        idx = rng.permutation(n) if shuffle_seed is not None else np.arange(n)
        for start in range(0, n - batch_size + 1, batch_size):
            batch_idx = idx[start : start + batch_size]
            examples = [normalize_row(dataset[int(i)]) for i in batch_idx]
            yield build_batch(examples, tokenizer, max_length)


def _to_text(x: object) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list) and x and isinstance(x[-1], dict) and "content" in x[-1]:
        return str(x[-1]["content"])
    raise ValueError(f"unrecognized completion format: {type(x)}")


def normalize_row(row: dict) -> dict:
    """UltraFeedback stores chosen/rejected as message lists; unwrap to plain strings."""
    prompt = row.get("prompt")
    chosen = row["chosen"]
    rejected = row["rejected"]

    if prompt is None:
        # Some datasets embed the prompt as the first message of chosen/rejected.
        if (
            isinstance(chosen, list)
            and chosen
            and isinstance(chosen[0], dict)
            and chosen[0].get("role") == "user"
        ):
            prompt = chosen[0]["content"]
        else:
            raise ValueError("dataset row has no `prompt` field and no user-turn fallback")

    return {"prompt": str(prompt), "chosen": _to_text(chosen), "rejected": _to_text(rejected)}
