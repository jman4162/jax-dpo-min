# jax-dpo-min

A minimal, from-scratch JAX implementation of **Direct Preference Optimization** (Rafailov et al., 2023). One loss, one model family, LoRA fine-tuning, one preference dataset, with ablations that probe whether the paper's objective is really doing what it claims.

The goal is correctness and legibility, not breadth. This is not a post-training framework — it is the DPO objective implemented cleanly enough to read.

## What's here

- `losses.py` — the DPO objective (plus `dpo_no_ref` and `dpo_no_weight` variants for ablations)
- `data.py` — tokenize `{prompt, chosen, rejected}` with a correct completion mask
- `model.py` — HuggingFace Flax model loading and sequence log-prob scoring
- `lora.py` — minimal LoRA adapter (single param tree serves both policy and frozen reference)
- `train.py` — single-device JIT-compiled training loop with Orbax checkpointing
- `eval.py` — pairwise preference accuracy on held-out pairs, plus a greedy sampler
- `configs/` — one laptop-CPU smoke recipe and one Colab-GPU headline recipe, plus four ablation configs
- `notebooks/01_dpo_from_paper.ipynb` — paper Eq. 7 → code walkthrough (runs standalone, no training required)
- `notebooks/02_evaluate_checkpoint.ipynb` — load a trained checkpoint and report pairwise preference accuracy (requires a prior `train.py` run)
- `colab/train_colab.ipynb` — Colab entry point for the headline recipe

## Two configs, one code path

| Config | Model | Hardware | Purpose |
|---|---|---|---|
| `configs/gpt2_cpu_smoke.yaml` | `gpt2` (124M) | Laptop CPU | Fast smoke test. Proves the code runs end-to-end and the loss decreases. **Not a chat-quality claim.** |
| `configs/gemma_colab.yaml` | `google/gemma-2b-it` | Colab T4/A100 | Headline recipe. Real instruct base, real preference data, measurable pairwise-accuracy gains. |
| `configs/gemma_tpu.yaml` | `google/gemma-2b-it` | Colab TPU v5e-1 / v6e-1 | Same recipe in bf16 for TPU matmul throughput. Single-core (no pmap — multi-device sharding is an explicit non-goal). |

Run the smoke config locally before burning a Colab session.

## Install

```bash
pip install -e ".[dev]"           # local / M-series Mac (CPU JAX)
pip install -e ".[dev,colab-gpu]" # Colab GPU runtime
```

Do not install `jax-metal`. Its Apple Silicon Metal backend is experimental; stick with CPU JAX locally.

## Run

```bash
pytest                                              # unit tests
python train.py --config configs/gpt2_cpu_smoke.yaml
python eval.py  --checkpoint outputs/gpt2_smoke/step_200
```

For Colab, open `colab/train_colab.ipynb` (GPU runtime) or `colab/train_tpu_colab.ipynb` (TPU runtime) and run all cells.

## Ablations

| Config | What changes | Expected behavior |
|---|---|---|
| `configs/gemma_colab.yaml` | — | Pairwise acc rises to ~0.65+ |
| `configs/abl_dpo_no_ref.yaml` | Drop the reference log-probs | Policy drifts; KL-to-ref large |
| `configs/abl_dpo_no_weight.yaml` | Remove the sigmoid weighting term | Training degenerates as the paper predicts |
| `configs/abl_beta_bad.yaml` | `beta = 10.0` | Worse or unstable training |
| `configs/abl_chosen_only.yaml` | SFT on chosen only | Fails to separate chosen from rejected |

Results are hand-written into `results/ablations.md` after each run.

## Non-goals

No PPO, no reward model, no multi-device sharding, no trainer framework. The deliberate scope is a clean DPO objective with ablations that probe the paper's claims — not a post-training platform.
