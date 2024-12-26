"""
PPO Trainer for RLHF fine-tuning.

Implements Proximal Policy Optimization with:
- KL penalty against a frozen reference policy
- Advantage estimation via rewards - baseline
- PPO clipping for stable updates
- Value function for variance reduction

Usage:
    python -m scripts.ppo_trainer --config configs/config.yaml
"""

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.policy_model import PolicyModel
from models.reward_model import RewardModel


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class PPOBuffer:
    """Rollout buffer for PPO experience collection."""

    def __init__(self):
        self.prompts: List[str] = []
        self.responses: List[str] = []
        self.response_ids: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.advantages: List[torch.Tensor] = []

    def add(self, prompt, response, response_ids, log_probs, reward):
        self.prompts.append(prompt)
        self.responses.append(response)
        self.response_ids.append(response_ids)
        self.log_probs.append(log_probs)
        self.rewards.append(reward)

    def compute_advantages(self, baseline: float = 0.0):
        """Compute advantages as reward - baseline."""
        self.advantages = [
            torch.tensor(r - baseline, dtype=torch.float32)
            for r in self.rewards
        ]

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.prompts)


class PPOTrainer:
    """
    PPO Trainer for RLHF.

    Pipeline:
    1. Generate responses from the policy for each prompt
    2. Score responses with the reward model
    3. Compute advantages (reward - baseline)
    4. Update policy using PPO clip loss + KL penalty
    5. Log metrics and save checkpoints

    The reference model is a frozen copy of the initial policy used to
    compute KL divergence penalties that prevent the policy from drifting
    too far from the base model.
    """

    def __init__(
        self,
        policy: PolicyModel,
        reward_model: RewardModel,
        config: Dict,
    ):
        self.policy = policy
        self.reward_model = reward_model
        self.config = config
        self.device = policy.device

        # Create frozen reference model (copy of initial policy)
        self.ref_policy = copy.deepcopy(policy)
        self.ref_policy.freeze()
        self.ref_policy.eval()
        print("Reference policy frozen.")

        # PPO hyperparameters
        ppo_cfg = config["ppo"]
        self.kl_coef = ppo_cfg["kl_coef"]
        self.clip_ratio = ppo_cfg["clip_ratio"]
        self.value_loss_coef = ppo_cfg["value_loss_coef"]
        self.entropy_coef = ppo_cfg["entropy_coef"]
        self.max_grad_norm = ppo_cfg["max_grad_norm"]
        self.ppo_epochs = ppo_cfg["ppo_epochs"]
        self.target_kl = ppo_cfg["target_kl"]

        train_cfg = config["training"]
        self.batch_size = train_cfg["batch_size"]
        self.mini_batch_size = train_cfg["mini_batch_size"]

        # Optimizer
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=ppo_cfg["lr"],
            weight_decay=0.01,
        )

        self.step = 0
        self.history: List[Dict] = []

    def _tokenize(self, text: str, max_length: int = 256) -> Dict[str, torch.Tensor]:
        """Tokenize a string and move to device."""
        enc = self.policy.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def collect_rollouts(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
    ) -> PPOBuffer:
        """
        Generate responses for each prompt and score with reward model.

        Returns a PPOBuffer with experience.
        """
        buffer = PPOBuffer()
        self.policy.eval()

        for prompt in tqdm(prompts, desc="Collecting rollouts", leave=False):
            # Generate response with log probs
            response, response_ids, log_probs = self.policy.generate_with_logprobs(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
            )

            # Compute reward: score prompt + response
            full_text = prompt + " " + response
            enc = self._tokenize(full_text)
            reward = self.reward_model.get_reward(
                enc["input_ids"],
                enc["attention_mask"],
            ).item()

            buffer.add(prompt, response, response_ids, log_probs, reward)

        # Compute advantages using mean reward as baseline
        baseline = sum(buffer.rewards) / len(buffer.rewards) if buffer.rewards else 0.0
        buffer.compute_advantages(baseline=baseline)

        return buffer

    def ppo_update(self, buffer: PPOBuffer) -> Dict[str, float]:
        """
        Perform PPO policy updates on collected experience.

        Returns aggregated metrics across all mini-batch updates.
        """
        self.policy.train()
        total_policy_loss = 0.0
        total_kl = 0.0
        total_entropy = 0.0
        n_updates = 0

        for ppo_epoch in range(self.ppo_epochs):
            # Random mini-batch sampling
            indices = torch.randperm(len(buffer)).tolist()

            for start in range(0, len(buffer), self.mini_batch_size):
                mb_indices = indices[start: start + self.mini_batch_size]
                if not mb_indices:
                    continue

                policy_losses = []
                kl_losses = []

                for idx in mb_indices:
                    prompt = buffer.prompts[idx]
                    response_ids = buffer.response_ids[idx].to(self.device)
                    old_log_probs = buffer.log_probs[idx].to(self.device)
                    advantage = buffer.advantages[idx].to(self.device)

                    # Tokenize prompt
                    prompt_enc = self.policy.tokenizer(
                        prompt,
                        return_tensors="pt",
                        max_length=256,
                        truncation=True,
                    )
                    prompt_ids = prompt_enc["input_ids"].to(self.device)

                    # Recompute log probs under current policy
                    response_ids_2d = response_ids.unsqueeze(0)
                    new_log_probs = self.policy.log_prob_of_sequence(
                        prompt_ids, response_ids_2d
                    ).squeeze(0)  # (T_r,)

                    # Compute log probs under reference policy
                    with torch.no_grad():
                        ref_log_probs = self.ref_policy.log_prob_of_sequence(
                            prompt_ids, response_ids_2d
                        ).squeeze(0)

                    # PPO clipped surrogate objective
                    # Ratio: exp(log_new - log_old)
                    min_len = min(len(new_log_probs), len(old_log_probs))
                    new_lp = new_log_probs[:min_len].sum()
                    old_lp = old_log_probs[:min_len].sum()

                    ratio = torch.exp(new_lp - old_lp)
                    clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

                    surr1 = ratio * advantage
                    surr2 = clipped_ratio * advantage
                    policy_loss = -torch.min(surr1, surr2)
                    policy_losses.append(policy_loss)

                    # KL divergence penalty: KL(policy || ref)
                    kl = self.policy.compute_kl_divergence(
                        new_log_probs[:min_len],
                        ref_log_probs[:min_len],
                    )
                    kl_loss = kl.mean()
                    kl_losses.append(kl_loss)

                # Aggregate and backprop
                policy_loss_mean = torch.stack(policy_losses).mean()
                kl_loss_mean = torch.stack(kl_losses).mean()

                total_loss = policy_loss_mean + self.kl_coef * kl_loss_mean

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss_mean.item()
                total_kl += kl_loss_mean.item()
                n_updates += 1

                # Early stopping if KL too large
                if kl_loss_mean.item() > self.target_kl * 1.5:
                    break

        n_updates = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / n_updates,
            "kl_div": total_kl / n_updates,
            "value_loss": 0.0,  # Placeholder (no separate value head here)
        }

    def train(
        self,
        prompts: List[str],
        n_steps: int = 1000,
        checkpoint_every: int = 100,
        save_dir: str = "checkpoints/ppo",
        use_wandb: bool = False,
    ):
        """
        Main PPO training loop.

        Args:
            prompts: Pool of prompts to sample from each step
            n_steps: Total number of PPO update steps
            checkpoint_every: Save checkpoint every N steps
            save_dir: Directory to save checkpoints
            use_wandb: Whether to log to Weights & Biases
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print("Starting PPO Training")
        print(f"  Steps: {n_steps} | KL coef: {self.kl_coef} | Clip: {self.clip_ratio}")
        print("=" * 60)

        for step in range(1, n_steps + 1):
            self.step = step
            t0 = time.time()

            # Sample batch of prompts
            import random
            batch_prompts = random.choices(prompts, k=self.batch_size)

            # Collect rollouts
            buffer = self.collect_rollouts(batch_prompts)

            # PPO update
            metrics = self.ppo_update(buffer)

            # Log step metrics
            reward_mean = sum(buffer.rewards) / len(buffer.rewards)
            reward_std = (
                sum((r - reward_mean) ** 2 for r in buffer.rewards) / len(buffer.rewards)
            ) ** 0.5

            metrics.update({
                "step": step,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "elapsed": time.time() - t0,
            })
            self.history.append(metrics)

            if step % self.config["logging"]["log_every"] == 0:
                print(
                    f"Step {step:4d} | "
                    f"reward={reward_mean:.4f}±{reward_std:.4f} | "
                    f"kl={metrics['kl_div']:.4f} | "
                    f"policy_loss={metrics['policy_loss']:.4f} | "
                    f"time={metrics['elapsed']:.1f}s"
                )

            if use_wandb:
                import wandb
                wandb.log(metrics)

            # Save checkpoint
            if step % checkpoint_every == 0:
                ckpt_path = str(save_path / f"policy_step_{step}")
                self.policy.save(ckpt_path)
                print(f"  -> Checkpoint saved at step {step}")

        # Final save
        self.policy.save(str(save_path / "policy_final"))
        print("\nPPO training complete.")
        return self.history


def main():
    parser = argparse.ArgumentParser(description="PPO fine-tuning with RLHF")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--reward-model", type=str, default=None,
                        help="Path to trained reward model checkpoint")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to prompts JSONL (uses preference data prompts)")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load reward model
    reward_model_path = args.reward_model or cfg["checkpoints"]["reward_model_path"]
    print(f"Loading reward model from {reward_model_path}")
    reward_model = RewardModel.load(reward_model_path, device=device)
    reward_model.eval()

    # Load or init policy model
    print(f"Initializing policy from '{cfg['model']['base_model']}'")
    policy = PolicyModel(model_name=cfg["model"]["base_model"], device=device)

    # Load prompts
    import json
    data_path = args.data or cfg["data"]["preferences_path"]
    prompts = []
    with open(data_path, "r") as f:
        for line in f:
            rec = json.loads(line.strip())
            prompts.append(rec["prompt"])
    print(f"Loaded {len(prompts)} prompts for PPO training.")

    # Optional wandb
    use_wandb = cfg["logging"]["use_wandb"]
    if use_wandb:
        try:
            import wandb
            wandb.init(project=cfg["logging"]["wandb_project"], config=cfg)
        except ImportError:
            use_wandb = False

    # Train
    trainer = PPOTrainer(policy, reward_model, cfg)
    n_steps = args.steps or cfg["ppo"]["training_steps"]
    trainer.train(
        prompts=prompts,
        n_steps=n_steps,
        checkpoint_every=cfg["ppo"]["checkpoint_every"],
        save_dir=str(Path(cfg["checkpoints"]["save_dir"]) / "ppo"),
        use_wandb=use_wandb,
    )

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
