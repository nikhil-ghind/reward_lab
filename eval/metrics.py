"""
Evaluation metrics for RLHF training.

Provides:
    - RewardModelAccuracy: percentage of pairs where chosen > rejected
    - KLDivergenceTracker: running mean KL divergence
    - RewardStats: mean/std/min/max reward over evaluation set
"""

import math
from typing import Dict, List, Optional

import torch
import numpy as np


class RewardModelAccuracy:
    """
    Tracks reward model accuracy: fraction of pairs where r_chosen > r_rejected.

    Usage:
        metric = RewardModelAccuracy()
        metric.update(chosen_rewards, rejected_rewards)
        acc = metric.compute()
    """

    def __init__(self):
        self.n_correct = 0
        self.n_total = 0

    def update(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ):
        """
        Update accuracy counter.

        Args:
            chosen_rewards: (B,) scalar rewards for chosen responses
            rejected_rewards: (B,) scalar rewards for rejected responses
        """
        with torch.no_grad():
            correct = (chosen_rewards > rejected_rewards).sum().item()
            self.n_correct += int(correct)
            self.n_total += chosen_rewards.numel()

    def compute(self) -> float:
        """Return accuracy as a float in [0, 1]."""
        if self.n_total == 0:
            return 0.0
        return self.n_correct / self.n_total

    def reset(self):
        """Reset counters."""
        self.n_correct = 0
        self.n_total = 0

    def __repr__(self) -> str:
        return (
            f"RewardModelAccuracy(accuracy={self.compute():.4f}, "
            f"n_correct={self.n_correct}, n_total={self.n_total})"
        )


class KLDivergenceTracker:
    """
    Tracks running mean of KL divergence between policy and reference.

    KL(policy || ref) = E[log p_policy - log p_ref]

    Usage:
        tracker = KLDivergenceTracker()
        tracker.update(kl_values)
        mean_kl = tracker.mean()
    """

    def __init__(self, window_size: Optional[int] = None):
        """
        Args:
            window_size: If set, computes running mean over last N values.
                         If None, computes mean over all values.
        """
        self.window_size = window_size
        self._values: List[float] = []

    def update(self, kl: torch.Tensor):
        """
        Add KL divergence values.

        Args:
            kl: Scalar or tensor of KL values (per token or per sequence)
        """
        if isinstance(kl, torch.Tensor):
            val = kl.detach().mean().item()
        else:
            val = float(kl)

        self._values.append(val)

        # Trim to window size
        if self.window_size is not None and len(self._values) > self.window_size:
            self._values = self._values[-self.window_size:]

    def mean(self) -> float:
        """Return running mean KL divergence."""
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    def std(self) -> float:
        """Return running std of KL divergence."""
        if len(self._values) < 2:
            return 0.0
        mean = self.mean()
        variance = sum((v - mean) ** 2 for v in self._values) / len(self._values)
        return math.sqrt(variance)

    def reset(self):
        """Reset tracker."""
        self._values.clear()

    def summary(self) -> Dict[str, float]:
        return {
            "kl_mean": self.mean(),
            "kl_std": self.std(),
            "kl_min": min(self._values) if self._values else 0.0,
            "kl_max": max(self._values) if self._values else 0.0,
            "n_updates": len(self._values),
        }

    def __repr__(self) -> str:
        return f"KLDivergenceTracker(mean={self.mean():.4f}, n={len(self._values)})"


class RewardStats:
    """
    Tracks reward statistics over an evaluation set.

    Computes mean, std, min, max reward.

    Usage:
        stats = RewardStats()
        stats.update(rewards_batch)
        summary = stats.compute()
    """

    def __init__(self):
        self._rewards: List[float] = []

    def update(self, rewards: torch.Tensor):
        """
        Add rewards from a batch.

        Args:
            rewards: (B,) reward tensor
        """
        with torch.no_grad():
            self._rewards.extend(rewards.detach().cpu().tolist())

    def compute(self) -> Dict[str, float]:
        """
        Compute reward statistics.

        Returns:
            Dict with keys: mean, std, min, max, n
        """
        if not self._rewards:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}

        arr = np.array(self._rewards)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": len(self._rewards),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
        }

    def reset(self):
        """Reset reward list."""
        self._rewards.clear()

    def __repr__(self) -> str:
        stats = self.compute()
        return (
            f"RewardStats(mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"min={stats['min']:.4f}, max={stats['max']:.4f}, n={stats['n']})"
        )
