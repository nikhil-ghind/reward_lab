"""
PreferenceDataset: loads JSONL preference data, tokenizes with HuggingFace tokenizer,
returns tensors for reward model training.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer


class PreferenceDataset(Dataset):
    """
    Dataset for reward model training from human preference data.

    Loads JSONL files with {prompt, chosen, rejected} format.
    Tokenizes prompt+chosen and prompt+rejected separately.
    Returns tensors suitable for Bradley-Terry reward model training.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "gpt2",
        max_length: int = 512,
        split: Optional[str] = None,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            data_path: Path to JSONL file with preference pairs
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length for tokenization
            split: 'train' or 'val' — if None, uses full dataset
            val_split: Fraction of data to use for validation
            seed: Random seed for train/val split
        """
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load all records
        records = self._load_jsonl(data_path)

        # Apply train/val split if requested
        if split is not None:
            n_val = max(1, int(len(records) * val_split))
            n_train = len(records) - n_val

            # Deterministic split using generator
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(len(records), generator=generator).tolist()
            if split == "train":
                records = [records[i] for i in indices[:n_train]]
            elif split == "val":
                records = [records[i] for i in indices[n_train:]]
            else:
                raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        self.records = records

    def _load_jsonl(self, path: str) -> List[Dict]:
        """Load all records from a JSONL file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        records = []
        with open(path, "r") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Validate required fields
                    for field in ("prompt", "chosen", "rejected"):
                        if field not in record:
                            raise ValueError(f"Missing field '{field}' on line {line_no}")
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Warning: skipping malformed JSON on line {line_no}: {e}")
                    continue

        return records

    def _tokenize_pair(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize a prompt+response pair.

        Returns dict with input_ids and attention_mask tensors.
        """
        text = prompt + " " + response
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with tokenized chosen and rejected sequences.

        Keys:
            chosen_input_ids: (max_length,)
            chosen_attention_mask: (max_length,)
            rejected_input_ids: (max_length,)
            rejected_attention_mask: (max_length,)
        """
        record = self.records[idx]
        prompt = record["prompt"]
        chosen = record["chosen"]
        rejected = record["rejected"]

        chosen_enc = self._tokenize_pair(prompt, chosen)
        rejected_enc = self._tokenize_pair(prompt, rejected)

        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
        }

    def get_prompts(self) -> List[str]:
        """Return list of all prompts (useful for PPO generation)."""
        return [r["prompt"] for r in self.records]


def create_dataloaders(
    data_path: str,
    tokenizer_name: str = "gpt2",
    max_length: int = 512,
    batch_size: int = 16,
    val_split: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function to create train and val DataLoaders.

    Args:
        data_path: Path to JSONL file
        tokenizer_name: HuggingFace tokenizer name
        max_length: Maximum sequence length
        batch_size: Batch size for DataLoader
        val_split: Validation fraction
        num_workers: Number of DataLoader workers
        seed: Random seed

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = PreferenceDataset(
        data_path=data_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        split="train",
        val_split=val_split,
        seed=seed,
    )

    val_dataset = PreferenceDataset(
        data_path=data_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        split="val",
        val_split=val_split,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Dataset sizes — train: {len(train_dataset)}, val: {len(val_dataset)}")
    return train_loader, val_loader
