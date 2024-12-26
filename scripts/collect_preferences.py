"""
CLI script for collecting preference data.

Usage:
    python -m scripts.collect_preferences --n-pairs 1000 --output data/preferences.jsonl
    python -m scripts.collect_preferences --n-pairs 500 --model gpt2 --seed 123
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preference_collector import PreferenceDataCollector


def main():
    parser = argparse.ArgumentParser(
        description="Collect preference pairs for RLHF training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=1000,
        help="Number of preference pairs to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/preferences.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="HuggingFace model name for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of pairs to process before flushing to disk",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Preference Data Collection")
    print("=" * 60)
    print(f"  Model:    {args.model}")
    print(f"  N Pairs:  {args.n_pairs}")
    print(f"  Output:   {args.output}")
    print(f"  Seed:     {args.seed}")
    print("=" * 60 + "\n")

    collector = PreferenceDataCollector(
        model_name=args.model,
        seed=args.seed,
    )

    pairs = collector.collect(
        n_pairs=args.n_pairs,
        output_path=args.output,
        batch_size=args.batch_size,
    )

    print("\nCollection complete.")
    stats = collector.get_statistics(pairs)
    print("\nStatistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<30} {v:.4f}")
        else:
            print(f"  {k:<30} {v}")


if __name__ == "__main__":
    main()
