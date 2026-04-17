# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

This repository currently contains **only a planning memo** (`background.md`). No source code, build system, tests, or configuration exist yet. The first coding task will be to scaffold the project described below.

## Project intent

`jax-dpo-min` is intended to be a **small, from-scratch JAX implementation of Direct Preference Optimization (DPO)** — not a full RLHF framework and not a broad post-training library. The design goal is a clean, inspectable, reproducible DPO implementation that demonstrates understanding of the paper's math and the modern JAX stack. See `background.md` for the full rationale.

## Planned stack

- **Plain JAX + Optax + Orbax** as the core stack (Optax for optimization, Orbax for checkpointing).
- **Flax NNX** (the newer simplified Flax API) optionally for model/module ergonomics.
- **LoRA** as the only parameter-efficient tuning path.
- **One small public instruct model** (e.g. a small Gemma) — not a model zoo.
- **One task** with clear evaluation, preferably math (GSM8K-style) or code, so both preference metrics and task accuracy can be reported.

## Planned layout

The memo specifies this file-by-file structure — follow it when scaffolding:

- `losses.py` — `dpo_loss(policy_logps, ref_logps, beta, chosen_mask, rejected_mask)`
- `train.py` — single-GPU/single-host training loop, optional gradient accumulation
- `data.py` — transforms `{prompt, chosen, rejected}` into packed token tensors
- `lora.py` — minimal adapter injection for attention/MLP blocks
- `eval.py` — pairwise preference accuracy, response sampling, task accuracy
- `notebooks/01_dpo_from_paper.ipynb` — derivation from paper objective to code
- `configs/gsm8k_dpo_tiny.yaml` — one reproducible recipe

## Core loss semantics (from the DPO paper)

Implement only:
- sequence log-prob scoring (policy and frozen reference)
- the DPO loss (paper Eq. 7; implementation sketch in Appendix B)
- metrics: pairwise accuracy, chosen-minus-rejected margin, KL-to-ref

The paper explicitly warns that the update is **not** merely "increase chosen, decrease rejected" — the weighting term matters, and a naïve form can degenerate. Preserve the weighting faithfully; an ablation that removes it is part of the planned deliverables (see below), but the default path must be the paper's form.

## Planned ablations (deliverable)

A single ablation table is part of the intended output and should be preserved when implementing:

- SFT only
- chosen-only fine-tune
- DPO (default)
- DPO without ref model, or with a bad `beta`
- DPO without the weighting term (to demonstrate why the paper's form matters)

## Explicit non-goals

Do **not** build these, even if they seem like natural extensions:

- A full PPO/RLHF stack
- A reward model trainer
- Distributed TPU orchestration
- A multi-algorithm trainer zoo
- An "alignment platform"

The project's value is in scope control. Tunix and EasyDeL already cover the broader post-training surface; competing on breadth is out of scope.
