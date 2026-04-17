"""DPO training loop: single-device, jit-compiled, LoRA-only grads.

One train step runs two forward passes — once with LoRA-merged params
(policy) and once with the base params alone (reference) — then computes
the DPO loss from their sequence log-probs and steps Optax AdamW on the
LoRA params only. The base model is never in the optimizer state.
"""
from __future__ import annotations

import argparse
import dataclasses
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from lora import count_params, init_lora, merge
from losses import dpo_loss
from model import LORA_TARGETS, compute_logps, load_model_and_tokenizer


@dataclasses.dataclass
class Config:
    model: str
    dataset: str
    max_length: int
    batch_size: int
    learning_rate: float
    beta: float
    lora_rank: int
    lora_alpha: int
    num_train_steps: int
    eval_every: int
    checkpoint_every: int
    seed: int
    loss_variant: str = "dpo"
    dataset_split: str = "train[:2000]"
    eval_split: str = "train[-256:]"
    dtype: str = "float32"


def load_config(path: str) -> Config:
    with open(path) as f:
        return Config(**yaml.safe_load(f))


def _stacked_forward(apply_fn, params, batch):
    """One forward pass over [chosen; rejected] stacked — halves the kernel launches."""
    input_ids = jnp.concatenate([batch["chosen_input_ids"], batch["rejected_input_ids"]], axis=0)
    attn = jnp.concatenate([batch["chosen_attention_mask"], batch["rejected_attention_mask"]], axis=0)
    cmask = jnp.concatenate([batch["chosen_completion_mask"], batch["rejected_completion_mask"]], axis=0)
    logps = compute_logps(apply_fn, params, input_ids, attn, cmask)
    b = batch["chosen_input_ids"].shape[0]
    return logps[:b], logps[b:]


def make_train_step(model, base_params, cfg: Config):
    optimizer = optax.adamw(cfg.learning_rate)

    def loss_fn(lora_params, batch):
        merged = merge(base_params, lora_params, alpha=cfg.lora_alpha, rank=cfg.lora_rank)
        pol_c, pol_r = _stacked_forward(model, merged, batch)
        ref_c, ref_r = _stacked_forward(model, base_params, batch)
        ref_c = jax.lax.stop_gradient(ref_c)
        ref_r = jax.lax.stop_gradient(ref_r)
        loss, metrics = dpo_loss(
            pol_c, pol_r, ref_c, ref_r, beta=cfg.beta, loss_variant=cfg.loss_variant
        )
        metrics = {**metrics, "kl_to_ref": (pol_c - ref_c).mean()}
        return loss, metrics

    @jax.jit
    def train_step(lora_params, opt_state, batch):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(lora_params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, lora_params)
        lora_params = optax.apply_updates(lora_params, updates)
        metrics = {**metrics, "loss": loss}
        return lora_params, opt_state, metrics

    return optimizer, train_step


def _evaluate(model, base_params, lora_params, cfg: Config, eval_batches) -> float:
    accs = []
    for batch in eval_batches:
        batch = {k: jnp.asarray(v) for k, v in batch.items()}
        merged = merge(base_params, lora_params, alpha=cfg.lora_alpha, rank=cfg.lora_rank)
        pol_c, pol_r = _stacked_forward(model, merged, batch)
        ref_c, ref_r = _stacked_forward(model, base_params, batch)
        cr = cfg.beta * (pol_c - ref_c)
        rr = cfg.beta * (pol_r - ref_r)
        accs.append(float((cr > rr).astype(jnp.float32).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def _save_checkpoint(out_dir: Path, step: int, lora_params, opt_state, cfg: Config):
    ckpt_dir = out_dir / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "state.pkl", "wb") as f:
        pickle.dump(
            {
                "step": step,
                "lora_params": jax.tree_util.tree_map(np.asarray, lora_params),
                "opt_state": jax.tree_util.tree_map(np.asarray, opt_state),
                "config": dataclasses.asdict(cfg),
            },
            f,
        )


def train(cfg: Config, output_dir: str):
    from datasets import load_dataset

    from data import build_batch, iter_batches, normalize_row

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] devices: {jax.devices()}")
    print(f"[info] loading {cfg.model} (dtype={cfg.dtype})")
    model, base_params, tokenizer = load_model_and_tokenizer(cfg.model, dtype=cfg.dtype)

    target_modules = LORA_TARGETS.get(cfg.model)
    if target_modules is None:
        raise ValueError(f"no LoRA target modules known for {cfg.model}; add an entry to model.LORA_TARGETS")
    print(f"[info] initializing LoRA rank={cfg.lora_rank} targets={target_modules}")
    lora_params = init_lora(base_params, target_modules, cfg.lora_rank, cfg.lora_alpha, seed=cfg.seed)
    print(f"[info] trainable LoRA params: {count_params(lora_params):,}")

    optimizer, train_step = make_train_step(model, base_params, cfg)
    opt_state = optimizer.init(lora_params)

    print(f"[info] loading dataset {cfg.dataset}")
    train_ds = load_dataset(cfg.dataset, split=cfg.dataset_split)
    eval_ds = load_dataset(cfg.dataset, split=cfg.eval_split)

    batch_iter = iter_batches(train_ds, tokenizer, cfg.batch_size, cfg.max_length, shuffle_seed=cfg.seed)

    def _fixed_eval_batches():
        batches = []
        for start in range(0, len(eval_ds) - cfg.batch_size + 1, cfg.batch_size):
            exs = [normalize_row(eval_ds[int(i)]) for i in range(start, start + cfg.batch_size)]
            batches.append(build_batch(exs, tokenizer, cfg.max_length))
        return batches

    eval_batches = _fixed_eval_batches()

    t0 = time.time()
    for step in range(1, cfg.num_train_steps + 1):
        batch = next(batch_iter)
        batch = {k: jnp.asarray(v) for k, v in batch.items()}
        lora_params, opt_state, metrics = train_step(lora_params, opt_state, batch)

        if step % cfg.eval_every == 0 or step == 1:
            acc = _evaluate(model, base_params, lora_params, cfg, eval_batches)
            elapsed = time.time() - t0
            print(
                f"[step {step:>6}] loss={float(metrics['loss']):.4f} "
                f"margin={float(metrics['margin']):+.3f} "
                f"train_acc={float(metrics['pairwise_accuracy']):.3f} "
                f"eval_acc={acc:.3f} "
                f"kl_ref={float(metrics['kl_to_ref']):+.3f} "
                f"elapsed={elapsed:.1f}s"
            )

        if step % cfg.checkpoint_every == 0:
            _save_checkpoint(out_dir, step, lora_params, opt_state, cfg)
            print(f"[ckpt] saved step {step} to {out_dir}")

    _save_checkpoint(out_dir, cfg.num_train_steps, lora_params, opt_state, cfg)
    print("[done]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    out = args.output_dir or f"outputs/{Path(args.config).stem}"
    train(cfg, out)


if __name__ == "__main__":
    main()
