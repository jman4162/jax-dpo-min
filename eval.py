"""Evaluation: pairwise preference accuracy on a held-out split, plus a
greedy sampler for manual inspection.

Pairwise accuracy is the primary metric — it's cheap (two forward passes per
batch, same as training), directly measures the DPO objective, and doesn't
depend on a judge model.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from data import build_batch, normalize_row
from lora import merge
from model import compute_logps, load_model_and_tokenizer


def load_checkpoint(path: str):
    with open(Path(path) / "state.pkl", "rb") as f:
        return pickle.load(f)


def _stacked_logps(model, params, batch):
    ids = jnp.concatenate([batch["chosen_input_ids"], batch["rejected_input_ids"]], axis=0)
    am = jnp.concatenate([batch["chosen_attention_mask"], batch["rejected_attention_mask"]], axis=0)
    cm = jnp.concatenate([batch["chosen_completion_mask"], batch["rejected_completion_mask"]], axis=0)
    logps = compute_logps(model, params, ids, am, cm)
    b = batch["chosen_input_ids"].shape[0]
    return logps[:b], logps[b:]


def pairwise_accuracy(model, base_params, lora_params, tokenizer, dataset, cfg) -> float:
    accs = []
    bs = cfg["batch_size"]
    max_len = cfg["max_length"]
    alpha = cfg["lora_alpha"]
    rank = cfg["lora_rank"]
    merged = merge(base_params, lora_params, alpha=alpha, rank=rank)
    for start in range(0, len(dataset) - bs + 1, bs):
        exs = [normalize_row(dataset[int(i)]) for i in range(start, start + bs)]
        batch = build_batch(exs, tokenizer, max_len)
        batch = {k: jnp.asarray(v) for k, v in batch.items()}
        pol_c, pol_r = _stacked_logps(model, merged, batch)
        ref_c, ref_r = _stacked_logps(model, base_params, batch)
        cr = pol_c - ref_c
        rr = pol_r - ref_r
        accs.append(float((cr > rr).astype(jnp.float32).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def greedy_sample(model, params, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    ids = tokenizer.encode(prompt, return_tensors="np")
    ids = jnp.asarray(ids)
    for _ in range(max_new_tokens):
        logits = model(input_ids=ids, params=params).logits
        next_id = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        ids = jnp.concatenate([ids, next_id], axis=-1)
        if int(next_id[0, 0]) == tokenizer.eos_token_id:
            break
    return tokenizer.decode(np.asarray(ids[0]), skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to a step_XXXXXX directory")
    p.add_argument("--dataset", default=None)
    p.add_argument("--split", default="train[-256:]")
    p.add_argument("--sample-prompt", default=None)
    args = p.parse_args()

    state = load_checkpoint(args.checkpoint)
    cfg = state["config"]
    lora_params = jax.tree_util.tree_map(jnp.asarray, state["lora_params"])

    model, base_params, tokenizer = load_model_and_tokenizer(cfg["model"], dtype=cfg.get("dtype", "float32"))

    if args.sample_prompt is not None:
        merged = merge(base_params, lora_params, alpha=cfg["lora_alpha"], rank=cfg["lora_rank"])
        out = greedy_sample(model, merged, tokenizer, args.sample_prompt)
        print(out)
        return

    from datasets import load_dataset

    ds_name = args.dataset or cfg["dataset"]
    ds = load_dataset(ds_name, split=args.split)
    acc = pairwise_accuracy(model, base_params, lora_params, tokenizer, ds, cfg)
    print(f"pairwise_accuracy: {acc:.4f}  (n={len(ds)})")


if __name__ == "__main__":
    main()
