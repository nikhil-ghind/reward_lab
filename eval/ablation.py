"""
Ablation study: vary kl_coef and measure reward vs KL tradeoff.

Runs short PPO training (50 steps) for each kl_coef value in
[0.01, 0.05, 0.1, 0.2, 0.5] and plots the reward vs kl_coef curve.

Usage:
    python -m eval.ablation --config configs/config.yaml --data data/preferences.jsonl
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving to file
import matplotlib.pyplot as plt
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.policy_model import PolicyModel
from models.reward_model import RewardModel
from scripts.ppo_trainer import PPOTrainer


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_ablation_trial(
    kl_coef: float,
    policy: PolicyModel,
    reward_model: RewardModel,
    prompts: List[str],
    base_config: Dict,
    n_steps: int = 50,
    batch_size: int = 4,
) -> Tuple[float, float]:
    """
    Run a single ablation trial with a given kl_coef.

    Returns:
        (final_reward_mean, final_kl_mean)
    """
    # Deep copy config and override kl_coef
    cfg = copy.deepcopy(base_config)
    cfg["ppo"]["kl_coef"] = kl_coef
    cfg["training"]["batch_size"] = batch_size
    cfg["logging"]["log_every"] = 9999  # suppress per-step logging

    # Deep copy policy to avoid contaminating across trials
    import copy as _copy
    trial_policy = _copy.deepcopy(policy)

    trainer = PPOTrainer(trial_policy, reward_model, cfg)

    import random
    rewards_per_step = []
    kl_per_step = []

    for step in range(1, n_steps + 1):
        batch_prompts = random.choices(prompts, k=batch_size)
        buffer = trainer.collect_rollouts(batch_prompts, max_new_tokens=64)
        metrics = trainer.ppo_update(buffer)

        reward_mean = sum(buffer.rewards) / max(len(buffer.rewards), 1)
        rewards_per_step.append(reward_mean)
        kl_per_step.append(metrics["kl_div"])

    # Use average of last 10 steps
    final_reward = sum(rewards_per_step[-10:]) / len(rewards_per_step[-10:])
    final_kl = sum(kl_per_step[-10:]) / len(kl_per_step[-10:])

    return final_reward, final_kl


def run_ablation(
    config_path: str,
    data_path: str,
    reward_model_path: str,
    kl_coefs: List[float] = None,
    n_steps: int = 50,
    output_dir: str = "eval",
    device: str = "cpu",
):
    """
    Full ablation study over kl_coef values.

    Args:
        config_path: Path to YAML config
        data_path: Path to JSONL preference data
        reward_model_path: Path to trained reward model checkpoint
        kl_coefs: List of kl_coef values to test
        n_steps: PPO steps per trial
        output_dir: Directory to save results and plot
        device: Device to run on
    """
    if kl_coefs is None:
        kl_coefs = [0.01, 0.05, 0.1, 0.2, 0.5]

    cfg = load_config(config_path)

    # Override device
    cfg["training"]["device"] = device

    # Load data
    prompts = []
    with open(data_path, "r") as f:
        for line in f:
            rec = json.loads(line.strip())
            prompts.append(rec["prompt"])
    print(f"Loaded {len(prompts)} prompts.")

    # Load reward model
    print(f"Loading reward model from {reward_model_path}")
    reward_model = RewardModel.load(reward_model_path, device=device)
    reward_model.eval()

    # Initialize base policy
    base_policy = PolicyModel(
        model_name=cfg["model"]["base_model"],
        device=device,
    )

    # Run ablation
    results = []
    print(f"\nRunning ablation over kl_coef values: {kl_coefs}")
    print(f"Steps per trial: {n_steps}\n")

    for i, kl_coef in enumerate(kl_coefs):
        print(f"[{i+1}/{len(kl_coefs)}] kl_coef = {kl_coef}")
        reward, kl = run_ablation_trial(
            kl_coef=kl_coef,
            policy=base_policy,
            reward_model=reward_model,
            prompts=prompts,
            base_config=cfg,
            n_steps=n_steps,
            batch_size=min(cfg["training"]["batch_size"], 4),
        )
        results.append({
            "kl_coef": kl_coef,
            "final_reward": reward,
            "final_kl": kl,
        })
        print(f"  -> reward={reward:.4f}, kl={kl:.4f}")

    # Save results as JSON
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_path = str(output_path / "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plot
    plot_path = str(output_path / "ablation_kl.png")
    plot_ablation_results(results, plot_path)
    print(f"Plot saved to {plot_path}")

    return results


def plot_ablation_results(results: List[Dict], save_path: str):
    """
    Plot reward vs kl_coef tradeoff curve (dual y-axis).
    Saved as PNG.
    """
    kl_coefs = [r["kl_coef"] for r in results]
    rewards = [r["final_reward"] for r in results]
    kl_divs = [r["final_kl"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_reward = "#2196F3"
    color_kl = "#F44336"

    ax1.set_xlabel("KL Coefficient (β)", fontsize=12)
    ax1.set_ylabel("Final Reward", color=color_reward, fontsize=12)
    ax1.plot(kl_coefs, rewards, "o-", color=color_reward, linewidth=2,
             markersize=8, label="Reward")
    ax1.tick_params(axis="y", labelcolor=color_reward)

    ax2 = ax1.twinx()
    ax2.set_ylabel("KL Divergence", color=color_kl, fontsize=12)
    ax2.plot(kl_coefs, kl_divs, "s--", color=color_kl, linewidth=2,
             markersize=8, label="KL Divergence")
    ax2.tick_params(axis="y", labelcolor=color_kl)

    # Annotate optimal point (max reward)
    best_idx = rewards.index(max(rewards))
    ax1.annotate(
        f"Best: β={kl_coefs[best_idx]}",
        xy=(kl_coefs[best_idx], rewards[best_idx]),
        xytext=(kl_coefs[best_idx] + 0.02, rewards[best_idx] * 1.05),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    plt.title("RLHF Ablation: Reward vs KL Coefficient", fontsize=14)
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run KL coefficient ablation study for RLHF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data", type=str, default="data/preferences.jsonl")
    parser.add_argument("--reward-model", type=str,
                        default="checkpoints/reward_model_best.pt")
    parser.add_argument("--n-steps", type=int, default=50,
                        help="PPO steps per ablation trial")
    parser.add_argument("--output-dir", type=str, default="eval")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--kl-coefs",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.1, 0.2, 0.5],
        help="KL coefficient values to test",
    )
    args = parser.parse_args()

    results = run_ablation(
        config_path=args.config,
        data_path=args.data,
        reward_model_path=args.reward_model,
        kl_coefs=args.kl_coefs,
        n_steps=args.n_steps,
        output_dir=args.output_dir,
        device=args.device,
    )

    print("\nAblation Summary:")
    print(f"{'kl_coef':>10} | {'reward':>10} | {'kl_div':>10}")
    print("-" * 36)
    for r in results:
        print(f"{r['kl_coef']:>10.3f} | {r['final_reward']:>10.4f} | {r['final_kl']:>10.4f}")


if __name__ == "__main__":
    main()
