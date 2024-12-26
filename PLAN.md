# Reward Lab

## Project Overview

An end-to-end Reinforcement Learning from Human Feedback (RLHF) pipeline that: (1) collects human preference pairs from a UI or data file, (2) trains a reward model on those pairs using a Bradley-Terry loss, (3) fine-tunes a small LLM (e.g., GPT-2 or LLaMA-3 1B) using Proximal Policy Optimization (PPO) with KL-divergence penalties relative to the frozen reference policy, and (4) runs ablation studies comparing reward model sizes and KL coefficients, with all experiments tracked in Weights & Biases.

**Key Goals:**
- Build a complete RLHF pipeline from preference data to fine-tuned model
- Train a reward model that correctly scores completions by human preference
- Fine-tune a base LLM with PPO while keeping behavior close to the reference policy via KL penalty
- Track all metrics (reward, KL-divergence, policy loss, value loss) in Weights & Biases
- Run ablation studies over KL coefficient (beta) values and reward model architectures

---

## Tech Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.11+ |
| Deep Learning | PyTorch | 2.3.x |
| Transformers | Hugging Face Transformers | 4.41.x |
| RLHF / PPO | TRL (Transformer Reinforcement Learning) | 0.9.x |
| PEFT / LoRA | PEFT | 0.11.x |
| Experiment Tracking | Weights & Biases | 0.17.x |
| Data Processing | datasets (HuggingFace) | 2.x |
| Config | Hydra | 1.3.x |
| Accelerate | Hugging Face Accelerate | 0.30.x |
| Testing | pytest | 8.x |

---

## Architecture Overview

```
Human Preference Data
(prompt, chosen, rejected)
         |
         v
   Reward Model Training
   (Bradley-Terry loss, cross-entropy on pairs)
         |
         v
   Trained Reward Model
         |
         v
   PPO Fine-tuning Loop
   ┌─────────────────────┐
   │  Actor Policy (LLM) │
   │  Critic (Value Head) │
   │  Reference Policy   │  <-- frozen copy of base LLM
   │  Reward Model       │  <-- frozen after RM training
   └─────────────────────┘
         |
         v
   Fine-tuned LLM
         |
         v
   W&B Experiment Dashboard
   (reward curves, KL penalty, ablation comparison)
```

**Key Training Loop (PPO):**
1. Sample prompts from dataset
2. Generate completions with actor policy
3. Score completions with reward model
4. Compute KL penalty: `KL(actor || ref)` per token
5. Compute advantages using GAE
6. Update actor policy + value head via PPO clipped objective
7. Log all metrics to W&B

---

## Phase 1: Project Scaffolding

**Goal:** Set up project layout, dependencies, and Hydra config.

### Tasks

1. Create directory structure:
   ```
   rlhfRewardModelTrainer/
   ├── src/
   │   ├── __init__.py
   │   ├── data/
   │   │   ├── __init__.py
   │   │   ├── preference_dataset.py
   │   │   └── data_utils.py
   │   ├── reward_model/
   │   │   ├── __init__.py
   │   │   ├── model.py
   │   │   └── trainer.py
   │   ├── ppo/
   │   │   ├── __init__.py
   │   │   ├── trainer.py
   │   │   └── rollout.py
   │   └── utils/
   │       ├── __init__.py
   │       ├── logging_utils.py
   │       └── model_utils.py
   ├── configs/
   │   ├── config.yaml
   │   ├── reward_model/
   │   │   └── default.yaml
   │   └── ppo/
   │       └── default.yaml
   ├── scripts/
   │   ├── train_reward_model.py
   │   ├── train_ppo.py
   │   └── run_ablations.py
   ├── tests/
   │   ├── test_preference_dataset.py
   │   ├── test_reward_model.py
   │   └── test_ppo_trainer.py
   ├── requirements.txt
   └── .env.example
   ```

2. Create `requirements.txt`:
   ```
   torch==2.3.*
   transformers==4.41.*
   trl==0.9.*
   peft==0.11.*
   datasets==2.*
   wandb==0.17.*
   accelerate==0.30.*
   hydra-core==1.3.*
   omegaconf==2.3.*
   numpy==1.*
   tqdm==4.*
   pytest==8.*
   ```

3. Create `configs/config.yaml`:
   ```yaml
   defaults:
     - reward_model: default
     - ppo: default
   
   base_model: "gpt2"
   tokenizer: "gpt2"
   seed: 42
   output_dir: "./outputs"
   wandb_project: "rlhf-experiments"
   ```

4. Create `configs/reward_model/default.yaml`:
   ```yaml
   epochs: 3
   batch_size: 16
   lr: 1e-5
   weight_decay: 0.01
   max_length: 512
   hidden_dropout: 0.1
   save_path: "./outputs/reward_model"
   ```

5. Create `configs/ppo/default.yaml`:
   ```yaml
   epochs: 1
   batch_size: 16
   mini_batch_size: 4
   ppo_epochs: 4
   learning_rate: 1.41e-5
   kl_penalty: "kl"
   init_kl_coef: 0.2          # beta
   target_kl: 6.0
   clip_range: 0.2
   vf_coef: 0.1
   max_new_tokens: 128
   temperature: 1.0
   ```

6. Run: `pip install -r requirements.txt`

---

## Phase 2: Preference Dataset

**Goal:** Load and preprocess human preference pairs.

### Tasks

1. **`src/data/preference_dataset.py`**
   - Define `PreferenceSample` dataclass: `prompt: str`, `chosen: str`, `rejected: str`
   - Define `class PreferenceDataset(Dataset)` (inherits `torch.utils.data.Dataset`):
     - `__init__(self, samples: list[PreferenceSample], tokenizer, max_length: int = 512)`
     - `__len__` and `__getitem__` implemented
     - `__getitem__(self, idx) -> dict`: tokenizes `prompt + chosen` as `chosen_ids` and `prompt + rejected` as `rejected_ids`, both padded/truncated to `max_length`, returns `{"chosen_input_ids": Tensor, "chosen_attention_mask": Tensor, "rejected_input_ids": Tensor, "rejected_attention_mask": Tensor}`
   - Define `load_preference_data(filepath: str) -> list[PreferenceSample]`:
     - Reads a JSON Lines file where each line is `{"prompt": str, "chosen": str, "rejected": str}`
     - Returns list of `PreferenceSample`
   - Define `load_hf_preference_data(dataset_name: str, split: str) -> list[PreferenceSample]`:
     - Loads from HuggingFace Hub (e.g., `"Anthropic/hh-rlhf"`) using `datasets.load_dataset`
     - Maps HF dataset columns to `PreferenceSample` fields

2. **`src/data/data_utils.py`**
   - Define `split_train_val(samples: list[PreferenceSample], val_ratio: float = 0.1, seed: int = 42) -> tuple[list, list]`: random split
   - Define `collate_preference_batch(batch: list[dict]) -> dict`: custom collate function for DataLoader

3. **`tests/test_preference_dataset.py`**
   - Test `load_preference_data()` with a temp JSONL file of 5 records
   - Test `PreferenceDataset.__getitem__()` returns correct keys and tensor shapes
   - Test `split_train_val()` produces correct split sizes

---

## Phase 3: Reward Model

**Goal:** Define and train the reward model using Bradley-Terry pairwise loss.

### Tasks

1. **`src/reward_model/model.py`**
   - Define `class RewardModel(nn.Module)`:
     - `__init__(self, base_model_name: str, dropout: float = 0.1)`:
       - Loads `AutoModel.from_pretrained(base_model_name)` as `self.backbone`
       - Adds `self.reward_head = nn.Linear(hidden_size, 1)` where `hidden_size` is taken from `backbone.config.hidden_size`
       - Adds `self.dropout = nn.Dropout(dropout)`
     - `forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor`:
       - Passes inputs through backbone, takes `last_hidden_state[:, -1, :]` (last token representation for causal LMs, or `[CLS]` token for BERT-style)
       - Applies dropout, passes through `reward_head`
       - Returns scalar reward per sample, shape `(batch_size, 1)`
   - Define `class RewardModelWithValueHead` — an alias for the above, for compatibility with TRL

2. **`src/reward_model/trainer.py`**
   - Define `class RewardModelTrainer`:
     - `__init__(self, model: RewardModel, train_dataset: PreferenceDataset, val_dataset: PreferenceDataset, cfg, wandb_run)`
     - `def compute_loss(self, chosen_rewards: Tensor, rejected_rewards: Tensor) -> Tensor`:
       - Bradley-Terry loss: `loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()`
     - `def train(self)`:
       - Initializes `AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)`
       - Runs `cfg.epochs` epochs; each epoch: iterate DataLoader, forward pass chosen+rejected separately, compute loss, backward, step
       - Logs to W&B every 10 steps: `{"rm/train_loss": loss, "rm/accuracy": (chosen > rejected).float().mean()}`
       - Runs validation at end of each epoch: computes val loss and accuracy, logs `{"rm/val_loss": ..., "rm/val_accuracy": ...}`
       - Saves best model checkpoint to `cfg.save_path` based on val accuracy
     - `def evaluate(self, dataset: PreferenceDataset) -> dict`: returns `{"loss": float, "accuracy": float}`

3. **`tests/test_reward_model.py`**
   - Test `RewardModel.forward()` with a batch of 4 returns shape `(4, 1)`
   - Test `compute_loss()` returns a scalar tensor with `requires_grad=True`
   - Test that loss decreases after one gradient step on a small synthetic batch

---

## Phase 4: PPO Fine-tuning

**Goal:** Fine-tune the base LLM using PPO with the trained reward model as the reward signal.

### Tasks

1. **`src/ppo/rollout.py`**
   - Define `class RolloutBuffer`:
     - Stores: `queries` (list of token tensors), `responses` (list of token tensors), `rewards` (list of floats), `logprobs` (list of tensors), `values` (list of tensors), `advantages` (list of tensors), `returns` (list of tensors)
     - `def add(self, query, response, reward, logprob, value)`: appends to lists
     - `def compute_advantages(self, gamma: float = 1.0, lam: float = 0.95)`: runs GAE (Generalized Advantage Estimation) over stored values and rewards
     - `def clear(self)`: resets all lists

2. **`src/ppo/trainer.py`**
   - Use TRL's `PPOTrainer` as the base — extend or configure it:
   - Define `class RLHFPPOTrainer`:
     - `__init__(self, cfg, base_model_name: str, reward_model: RewardModel, tokenizer, wandb_run)`:
       - Loads actor model: `AutoModelForCausalLMWithValueHead.from_pretrained(base_model_name)` (from TRL)
       - Loads reference model: `AutoModelForCausalLM.from_pretrained(base_model_name)` with all params frozen (`requires_grad_(False)`)
       - Initializes TRL's `PPOConfig` from `cfg`
       - Initializes TRL's `PPOTrainer(config, actor_model, ref_model, tokenizer)`
       - Stores `reward_model` (frozen during PPO)
     - `def score_responses(self, queries: list[str], responses: list[str]) -> list[float]`:
       - Tokenizes `query + response` pairs
       - Passes through `reward_model`, returns list of scalar rewards
     - `def train_step(self, batch: dict) -> dict`:
       - Generates responses: `PPOTrainer.generate(query_tensors, max_new_tokens=cfg.max_new_tokens)`
       - Scores responses: `score_responses(queries, responses)`
       - Computes KL penalty: `PPOTrainer` handles this internally with `init_kl_coef`
       - Calls `PPOTrainer.step(query_tensors, response_tensors, rewards)`
       - Returns stats dict with keys: `ppo/loss/policy`, `ppo/loss/value`, `ppo/mean_reward`, `ppo/mean_kl`, `ppo/kl_penalty`
     - `def train(self, prompt_dataset)`:
       - Iterates over `cfg.epochs` epochs, each batch calls `train_step`
       - Logs all stats to W&B
       - Saves checkpoint every 100 steps to `{cfg.output_dir}/checkpoint_step_{step}`

---

## Phase 5: Training Scripts

**Goal:** Create runnable training scripts for reward model and PPO.

### Tasks

1. **`scripts/train_reward_model.py`**
   - Uses `@hydra.main(config_path="../configs", config_name="config")` decorator
   - Steps:
     1. `wandb.init(project=cfg.wandb_project, config=OmegaConf.to_container(cfg))`
     2. Load tokenizer: `AutoTokenizer.from_pretrained(cfg.tokenizer)`
     3. Load preference data: `load_hf_preference_data("Anthropic/hh-rlhf", "train")`
     4. Split into train/val: `split_train_val(samples)`
     5. Create `PreferenceDataset` for train and val
     6. Init `RewardModel(cfg.base_model, cfg.reward_model.hidden_dropout)`
     7. Init and run `RewardModelTrainer`
     8. Save final model and tokenizer
   - Run: `python scripts/train_reward_model.py`

2. **`scripts/train_ppo.py`**
   - Uses `@hydra.main` decorator
   - Steps:
     1. `wandb.init(project=cfg.wandb_project, ...)`
     2. Load tokenizer
     3. Load trained reward model from `cfg.reward_model.save_path`
     4. Load prompt-only dataset (e.g., first 10k prompts from `"Anthropic/hh-rlhf"`, strip completions)
     5. Init `RLHFPPOTrainer`
     6. Run `trainer.train(prompt_dataset)`
   - Run: `python scripts/train_ppo.py`

---

## Phase 6: Ablation Studies

**Goal:** Systematically compare configurations and log results to W&B.

### Tasks

1. **`scripts/run_ablations.py`**
   - Define ablation grid:
     ```python
     KL_COEFS = [0.05, 0.1, 0.2, 0.4]
     REWARD_MODEL_SIZES = ["gpt2", "gpt2-medium"]
     ```
   - For each combination, launch a training run (can be subprocess or sequential):
     - Override Hydra config: `ppo.init_kl_coef={kl_coef} base_model={rm_size}`
     - Set `wandb_run_name = f"ablation_kl{kl_coef}_rm{rm_size}"`
     - Log to same W&B project for comparison
   - After all runs, use W&B API to pull final reward means and KL values:
     ```python
     import wandb
     api = wandb.Api()
     runs = api.runs("your_entity/rlhf-experiments", {"tags": {"$in": ["ablation"]}})
     ```
   - Print a summary table of `(kl_coef, rm_size) -> final_mean_reward, final_mean_kl`

2. **W&B Dashboard Setup:**
   - In each training run, log these metrics consistently:
     - `train/reward_mean`, `train/kl_mean`, `train/policy_loss`, `train/value_loss`
     - `eval/reward_mean`, `eval/kl_mean`
     - `config/kl_coef`, `config/rm_size`
   - Use W&B `sweep` or compare runs in the W&B UI via parallel coordinates plot

---

## Phase 7: Testing

**Goal:** Unit test key components without running full training.

### Tasks

1. **`tests/test_preference_dataset.py`**: (see Phase 2)

2. **`tests/test_reward_model.py`**: (see Phase 3)

3. **`tests/test_ppo_trainer.py`**
   - Use `gpt2` model with a tiny dataset of 2 prompts
   - Mock `RewardModel.forward()` to return constant rewards
   - Test `RLHFPPOTrainer.score_responses()` returns a list of floats with correct length
   - Test `train_step()` completes without error and returns expected stat keys
   - Use `torch.manual_seed(42)` for reproducibility

4. Run: `pytest tests/ -v -x`
