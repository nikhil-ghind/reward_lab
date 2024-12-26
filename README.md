# RewardLab

A complete Reinforcement Learning from Human Feedback (RLHF) pipeline in Python. Covers preference data collection, reward model training, PPO fine-tuning with KL penalties, and ablation studies.

## What is RLHF?

RLHF aligns language models with human preferences by training them using feedback from human evaluators. The pipeline consists of three stages:

```
┌──────────────────────────────────────────────────────────────┐
│                     RLHF Pipeline                            │
│                                                              │
│  1. COLLECT          2. REWARD MODEL       3. PPO TRAINING   │
│  ──────────         ──────────────────    ──────────────── │
│  Prompt model   →   Train on (chosen,  →  Optimize policy   │
│  twice, label       rejected) pairs       to maximize       │
│  preferred          Bradley-Terry loss    reward - β·KL     │
│  response                                                    │
└──────────────────────────────────────────────────────────────┘
```

## Project Structure

```
reward_lab/
├── configs/
│   └── config.yaml          # All hyperparameters
├── data/
│   ├── preference_collector.py  # Generate preference pairs
│   └── dataset.py               # PyTorch dataset for training
├── models/
│   ├── reward_model.py      # GPT-2 + scalar head, Bradley-Terry loss
│   └── policy_model.py      # Policy wrapper with log probs + KL
├── scripts/
│   ├── collect_preferences.py   # CLI: generate preference data
│   ├── train_reward_model.py    # Train the reward model
│   └── ppo_trainer.py           # Full PPO training loop
├── eval/
│   ├── metrics.py           # Accuracy, KL tracker, reward stats
│   └── ablation.py          # KL coefficient ablation study
├── notebooks/
│   └── rlhf_walkthrough.ipynb  # End-to-end walkthrough
└── requirements.txt
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect preference data

```bash
python -m scripts.collect_preferences \
    --n-pairs 1000 \
    --output data/preferences.jsonl \
    --model gpt2
```

### 3. Train the reward model

```bash
python -m scripts.train_reward_model \
    --config configs/config.yaml \
    --data data/preferences.jsonl
```

Expected output:
```
Epoch   1 | train_loss=0.6821 | train_acc=0.612 | val_loss=0.6734 | val_acc=0.623 | time=45.2s
Epoch   2 | train_loss=0.6105 | train_acc=0.693 | val_loss=0.6022 | val_acc=0.701 | time=44.8s
Epoch   3 | train_loss=0.5544 | train_acc=0.741 | val_loss=0.5601 | val_acc=0.738 | time=45.1s
```

### 4. PPO fine-tuning

```bash
python -m scripts.ppo_trainer \
    --config configs/config.yaml \
    --reward-model checkpoints/reward_model_best.pt \
    --data data/preferences.jsonl
```

Expected output:
```
Step  100 | reward=0.3241±0.1823 | kl=0.0412 | policy_loss=0.0231 | time=8.2s
Step  200 | reward=0.4127±0.1654 | kl=0.0891 | policy_loss=0.0198 | time=8.1s
```

### 5. Run KL coefficient ablation

```bash
python -m eval.ablation \
    --config configs/config.yaml \
    --data data/preferences.jsonl \
    --reward-model checkpoints/reward_model_best.pt \
    --kl-coefs 0.01 0.05 0.1 0.2 0.5 \
    --n-steps 50
```

Results saved to `eval/ablation_results.json` and `eval/ablation_kl.png`.

## Configuration Reference

All hyperparameters live in `configs/config.yaml`:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `model` | `base_model` | `gpt2` | HuggingFace model ID |
| `model` | `max_seq_len` | `512` | Maximum token sequence length |
| `reward_model` | `lr` | `1e-5` | Reward model learning rate |
| `reward_model` | `epochs` | `3` | Training epochs |
| `ppo` | `lr` | `1e-5` | Policy learning rate |
| `ppo` | `kl_coef` | `0.1` | KL divergence penalty coefficient (β) |
| `ppo` | `clip_ratio` | `0.2` | PPO surrogate clipping ratio (ε) |
| `ppo` | `training_steps` | `1000` | Total PPO update steps |
| `training` | `batch_size` | `16` | Batch size |
| `training` | `seed` | `42` | Random seed |

## Ablation Results Format

The ablation study produces `eval/ablation_results.json`:

```json
[
  {"kl_coef": 0.01, "final_reward": 0.412, "final_kl": 0.892},
  {"kl_coef": 0.05, "final_reward": 0.398, "final_kl": 0.421},
  {"kl_coef": 0.10, "final_reward": 0.371, "final_kl": 0.198},
  {"kl_coef": 0.20, "final_reward": 0.334, "final_kl": 0.089},
  {"kl_coef": 0.50, "final_reward": 0.281, "final_kl": 0.031}
]
```

The plot `eval/ablation_kl.png` shows the reward-KL tradeoff curve:
- Low β: high reward but high KL (policy drifts far from reference)
- High β: low KL but suppressed reward (policy stays too close to SFT)
- Optimal β ≈ 0.05–0.1 typically balances both objectives

## Expected Training Curves

**Reward model accuracy** should increase monotonically from ~50% (random) to 70–80% over 3 epochs.

**PPO reward** should increase over training steps, stabilizing after ~500 steps.

**KL divergence** should stay controlled (< `target_kl`) if `kl_coef` is tuned correctly.

## Key Design Decisions

- **Bradley-Terry loss**: Models pairwise preferences as `P(chosen > rejected) = σ(r_chosen - r_rejected)`. This is more stable than pointwise regression.
- **KL penalty**: Prevents reward hacking by penalizing large deviations from the reference policy: `reward_shaped = reward - β · KL(π || π_ref)`.
- **PPO clipping**: Limits the policy update ratio to `[1-ε, 1+ε]` for stable optimization.
- **Frozen reference**: A copy of the initial policy is frozen at the start of PPO and used throughout to compute KL.
