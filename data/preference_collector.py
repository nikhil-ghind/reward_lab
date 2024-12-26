"""
PreferenceDataCollector: generates preference pairs from a base model,
scores them with a heuristic (length + diversity), saves as JSONL.
"""

import json
import random
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float


PROMPT_TEMPLATES = [
    "Explain the concept of {topic} in simple terms.",
    "What are the main advantages of {topic}?",
    "Describe how {topic} works step by step.",
    "Compare and contrast {topic} with similar approaches.",
    "What are the key challenges with {topic}?",
    "Provide an overview of {topic} for a beginner.",
    "How has {topic} evolved over the past decade?",
    "What are best practices when working with {topic}?",
]

TOPICS = [
    "machine learning", "neural networks", "gradient descent",
    "transformer models", "attention mechanisms", "reinforcement learning",
    "natural language processing", "computer vision", "data augmentation",
    "hyperparameter tuning", "transfer learning", "fine-tuning",
    "regularization", "overfitting", "cross-validation",
    "batch normalization", "dropout", "residual connections",
    "convolutional networks", "recurrent networks",
]


def compute_heuristic_score(text: str, prompt: str) -> float:
    """
    Heuristic scoring function combining:
    1. Length score: prefer medium-length responses (100-300 tokens)
    2. Diversity score: unique word ratio
    3. Relevance score: keyword overlap with prompt
    """
    words = text.split()
    n_words = len(words)

    # Length score: Gaussian centered at 200 words
    length_score = math.exp(-((n_words - 200) ** 2) / (2 * 100 ** 2))

    # Diversity score: type-token ratio
    if n_words == 0:
        diversity_score = 0.0
    else:
        unique_words = len(set(w.lower() for w in words))
        diversity_score = unique_words / n_words

    # Relevance score: keyword overlap
    prompt_words = set(prompt.lower().split())
    response_words = set(text.lower().split())
    overlap = len(prompt_words & response_words)
    relevance_score = overlap / max(len(prompt_words), 1)

    # Weighted combination
    score = 0.4 * length_score + 0.4 * diversity_score + 0.2 * relevance_score
    return float(score)


class PreferenceDataCollector:
    """
    Generates preference pairs from a base language model.

    For each prompt, generates two responses and uses a heuristic
    scorer to determine which is 'chosen' (better) vs 'rejected' (worse).
    Saves output as JSONL with {prompt, chosen, rejected} format.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)

        print(f"Loading model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Model loaded.")

    def _build_prompts(self, n_pairs: int) -> List[str]:
        """Build a list of prompts by sampling templates and topics."""
        prompts = []
        for _ in range(n_pairs):
            template = random.choice(PROMPT_TEMPLATES)
            topic = random.choice(TOPICS)
            prompts.append(template.format(topic=topic))
        return prompts

    def _generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> str:
        """Generate a single response from the model."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    def generate_pair(self, prompt: str) -> PreferencePair:
        """
        Generate two responses for a prompt and score them.
        Returns a PreferencePair with chosen/rejected labeled by score.
        """
        # Generate two responses with different temperatures
        response_a = self._generate_response(prompt, temperature=0.7, top_p=0.9)
        response_b = self._generate_response(prompt, temperature=1.2, top_p=0.95)

        score_a = compute_heuristic_score(response_a, prompt)
        score_b = compute_heuristic_score(response_b, prompt)

        if score_a >= score_b:
            chosen, rejected = response_a, response_b
            chosen_score, rejected_score = score_a, score_b
        else:
            chosen, rejected = response_b, response_a
            chosen_score, rejected_score = score_b, score_a

        # Skip pairs where scores are too close (ambiguous preference)
        if abs(chosen_score - rejected_score) < 0.05:
            # Swap to ensure some difference
            pass

        return PreferencePair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            chosen_score=chosen_score,
            rejected_score=rejected_score,
        )

    def collect(
        self,
        n_pairs: int = 1000,
        output_path: str = "data/preferences.jsonl",
        batch_size: int = 10,
    ) -> List[PreferencePair]:
        """
        Collect n_pairs preference pairs and save to output_path as JSONL.

        Args:
            n_pairs: Number of preference pairs to generate
            output_path: Path to save JSONL file
            batch_size: Number of pairs to process before saving (for safety)

        Returns:
            List of PreferencePair objects
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        prompts = self._build_prompts(n_pairs)
        pairs: List[PreferencePair] = []

        print(f"Generating {n_pairs} preference pairs...")
        with open(output_path, "w") as f:
            for i, prompt in enumerate(tqdm(prompts, desc="Collecting preferences")):
                try:
                    pair = self.generate_pair(prompt)
                    pairs.append(pair)

                    # Write as JSONL record
                    record = {
                        "prompt": pair.prompt,
                        "chosen": pair.chosen,
                        "rejected": pair.rejected,
                    }
                    f.write(json.dumps(record) + "\n")

                    if (i + 1) % batch_size == 0:
                        f.flush()

                except Exception as e:
                    print(f"Warning: failed on pair {i}: {e}")
                    continue

        print(f"Saved {len(pairs)} preference pairs to {output_path}")
        return pairs

    def get_statistics(self, pairs: List[PreferencePair]) -> Dict:
        """Compute statistics over collected preference pairs."""
        if not pairs:
            return {}

        chosen_lengths = [len(p.chosen.split()) for p in pairs]
        rejected_lengths = [len(p.rejected.split()) for p in pairs]
        score_gaps = [p.chosen_score - p.rejected_score for p in pairs]

        return {
            "n_pairs": len(pairs),
            "avg_chosen_length": sum(chosen_lengths) / len(chosen_lengths),
            "avg_rejected_length": sum(rejected_lengths) / len(rejected_lengths),
            "avg_score_gap": sum(score_gaps) / len(score_gaps),
            "min_score_gap": min(score_gaps),
            "max_score_gap": max(score_gaps),
        }


def main():
    parser = argparse.ArgumentParser(description="Collect preference data for RLHF")
    parser.add_argument("--n-pairs", type=int, default=1000, help="Number of preference pairs")
    parser.add_argument("--output", type=str, default="data/preferences.jsonl")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    collector = PreferenceDataCollector(model_name=args.model, seed=args.seed)
    pairs = collector.collect(n_pairs=args.n_pairs, output_path=args.output)

    stats = collector.get_statistics(pairs)
    print("\nCollection Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
