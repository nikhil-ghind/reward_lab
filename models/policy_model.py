"""
PolicyModel: wraps AutoModelForCausalLM for PPO fine-tuning.
Provides generation with log probabilities and KL divergence computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple


class PolicyModel(nn.Module):
    """
    Policy model wrapper for PPO RLHF training.

    Wraps a causal language model and adds:
        - generate_with_logprobs: generation with token log probabilities
        - compute_kl_divergence: KL(policy || reference) estimation
        - log_prob_of_sequence: compute log probability of a given sequence

    The model is typically initialized from a supervised fine-tuned (SFT)
    checkpoint before PPO training begins. A frozen reference copy is kept
    to compute KL penalties.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass returning logits.

        Args:
            input_ids: (B, T) input token ids
            attention_mask: (B, T) attention mask
            labels: (B, T) target labels for loss computation

        Returns:
            logits: (B, T, vocab_size)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.logits

    def generate_with_logprobs(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Generate a response and return per-token log probabilities.

        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling threshold

        Returns:
            response: Generated text string (without prompt)
            response_ids: (T_new,) generated token ids
            log_probs: (T_new,) log probability of each generated token
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # output.sequences: (1, input_len + new_tokens)
        # output.scores: list of (1, vocab_size) logit tensors, one per new token
        response_ids = output.sequences[0][input_len:]  # (T_new,)

        # Compute log probs from scores
        # scores[t] is the logit distribution BEFORE sampling token t+1
        log_probs_list = []
        for t, score in enumerate(output.scores):
            # score: (1, vocab_size)
            log_prob_dist = F.log_softmax(score / temperature, dim=-1)  # (1, V)
            token_id = response_ids[t]
            log_prob = log_prob_dist[0, token_id]
            log_probs_list.append(log_prob)

        log_probs = torch.stack(log_probs_list)  # (T_new,)
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return response, response_ids, log_probs

    def log_prob_of_sequence(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities for a given response sequence.

        Used during PPO to recompute log probs under the current policy.

        Args:
            prompt_ids: (B, T_p) prompt token ids
            response_ids: (B, T_r) response token ids
            attention_mask: (B, T_p + T_r) combined attention mask

        Returns:
            log_probs: (B, T_r) per-token log probabilities
        """
        # Concatenate prompt and response
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)  # (B, T_p + T_r)

        if attention_mask is None:
            attention_mask = torch.ones_like(full_ids)

        logits = self.forward(full_ids, attention_mask)  # (B, T, V)

        # Log probs at positions corresponding to response tokens
        # We want log p(response[t] | prompt + response[:t])
        # This is logits at positions [T_p-1, T_p, ..., T_p+T_r-2]
        T_p = prompt_ids.shape[1]
        T_r = response_ids.shape[1]

        # logits for predicting response: positions T_p-1 through T_p+T_r-2
        response_logits = logits[:, T_p - 1: T_p + T_r - 1, :]  # (B, T_r, V)

        log_probs = F.log_softmax(response_logits, dim=-1)  # (B, T_r, V)

        # Gather the log prob of the actual response tokens
        # response_ids: (B, T_r) -> unsqueeze for gather
        token_log_probs = log_probs.gather(
            dim=-1,
            index=response_ids.unsqueeze(-1),
        ).squeeze(-1)  # (B, T_r)

        return token_log_probs

    def compute_kl_divergence(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate per-token KL divergence: KL(policy || reference).

        Uses the approximation: KL ≈ log_prob_policy - log_prob_ref

        Args:
            log_probs: (B, T) log probs under current policy
            ref_log_probs: (B, T) log probs under reference policy

        Returns:
            kl: (B, T) per-token KL divergence estimate
        """
        kl = log_probs - ref_log_probs
        return kl

    def get_value(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Placeholder value function (for full PPO, use a separate value head).
        Returns zeros — override with a proper value head for actor-critic PPO.
        """
        B = input_ids.shape[0]
        return torch.zeros(B, device=input_ids.device)

    def freeze(self):
        """Freeze all parameters (used for reference model)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def save(self, path: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Policy model saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "PolicyModel":
        """Load policy model from directory."""
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        obj.model_name = path
        obj.model = AutoModelForCausalLM.from_pretrained(path).to(obj.device)
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.tokenizer.pad_token = obj.tokenizer.eos_token
        return obj
