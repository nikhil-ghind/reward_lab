"""
RewardModel: GPT-2 backbone with a linear scalar head for RLHF reward modeling.
Trained with Bradley-Terry pairwise preference loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, Tuple


class RewardModel(nn.Module):
    """
    Reward model built on top of a pretrained language model backbone.

    Architecture:
        - GPT-2 (or any AutoModel) backbone
        - Linear head projecting from hidden_size -> 1 (scalar reward)

    Training objective:
        Bradley-Terry pairwise loss:
            L = -log(sigmoid(r_chosen - r_rejected))

    This encourages r_chosen > r_rejected for each preference pair.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        hidden_size = self.config.hidden_size

        # Scalar reward head
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        # Initialize reward head weights
        self._init_weights()

    def _init_weights(self):
        """Initialize reward head with small weights for stable training."""
        for module in self.reward_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scalar reward for each sequence in the batch.

        Args:
            input_ids: (batch_size, seq_len) token ids
            attention_mask: (batch_size, seq_len) mask (1=valid, 0=pad)

        Returns:
            rewards: (batch_size,) scalar reward values
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use the last non-padding token's hidden state as sequence representation
        # For GPT-2 (decoder-only), we use the last valid token
        hidden_states = outputs.last_hidden_state  # (B, T, H)

        if attention_mask is not None:
            # Find the index of the last valid token for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
            seq_lengths = seq_lengths.clamp(min=0)
            batch_size = hidden_states.size(0)
            # Gather last valid hidden state: (B, H)
            last_hidden = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                seq_lengths,
            ]
        else:
            last_hidden = hidden_states[:, -1, :]  # (B, H)

        rewards = self.reward_head(last_hidden).squeeze(-1)  # (B,)
        return rewards

    def compute_reward_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Bradley-Terry pairwise ranking loss.

        L = -log(sigmoid(r_chosen - r_rejected))

        Args:
            chosen_input_ids: (B, T) token ids for chosen responses
            chosen_attention_mask: (B, T) attention mask for chosen
            rejected_input_ids: (B, T) token ids for rejected responses
            rejected_attention_mask: (B, T) attention mask for rejected

        Returns:
            loss: scalar loss tensor
            metrics: dict with chosen/rejected reward stats
        """
        r_chosen = self.forward(chosen_input_ids, chosen_attention_mask)
        r_rejected = self.forward(rejected_input_ids, rejected_attention_mask)

        # Bradley-Terry loss: maximize margin between chosen and rejected
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()

        # Compute accuracy: how often is r_chosen > r_rejected?
        with torch.no_grad():
            accuracy = (r_chosen > r_rejected).float().mean().item()

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy,
            "chosen_reward_mean": r_chosen.mean().item(),
            "rejected_reward_mean": r_rejected.mean().item(),
            "reward_margin": (r_chosen - r_rejected).mean().item(),
        }

        return loss, metrics

    def get_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inference-time reward computation.

        Returns raw scalar rewards without gradient computation.
        """
        with torch.no_grad():
            return self.forward(input_ids, attention_mask)

    def save(self, path: str):
        """Save model state dict."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "model_name": self.config.name_or_path,
        }, path)
        print(f"Reward model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "RewardModel":
        """Load model from saved checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model_name = checkpoint.get("model_name", "gpt2")
        model = cls(model_name=model_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model
