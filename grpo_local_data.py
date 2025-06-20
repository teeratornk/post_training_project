# GRPO training script for Wordle using local CSV word list
# Based on basecase_l3_local_dataset.py, extended for GRPO training

import os
import re
import json
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import List
from logger_setup import logger
from sklearn.model_selection import train_test_split
from reward_functions import output_format_check, uses_previous_feedback, guess_value
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer
import torch
import torch.distributed as dist

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        logger.info(f"[RANK {os.environ['RANK']}] Using device: {device}")
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Distributed not initialized, using default device:", device)
    return local_rank, device

# ----------------------#
# 0.1 PROMPT TEMPLATES   #
# ----------------------#

SYSTEM_PROMPT = """
You are playing Wordle, a word-guessing game.

### Game Rules:
- You have **6 tries** to guess a secret **5-letter** word.
- Each guess must be a valid **5-letter English word**.
- After each guess, you will receive feedback indicating how close 
your guess was.

### Feedback Format:
Each letter in your guess will receive one of three symbols:
1. ✓ : The letter is in the word and in the CORRECT position.
2. - : The letter is in the word but in the WRONG position.
3. x : The letter is NOT in the word.

### Example:
Secret Word: BRISK

Guess 1: STORM → Feedback: S(-) T(x) O(x) R(-) M(x)
Guess 2: BRAVE → Feedback: B(✓) R(✓) A(x) V(x) E(x)
Guess 3: BRISK → Feedback: B(✓) R(✓) I(✓) S(✓) K(✓)

### Response Format:
Think through the problem and feedback step by step. Make sure to 
first add your step by step thought process within <think> </think> 
tags. Then, return your guessed word in the following format: 
<guess> guessed-word </guess>.
"""

# ----------------------#
# 0.2 PROMPT FUNCTIONS   #
# ----------------------#

# For RLHF, you can use a more informative prompt by incorporating render_prompt (with empty past_guesses for each sample)
def render_prompt_for_dataset():
    return SYSTEM_PROMPT + "\nMake a new 5-letter word guess.\nLet me solve this step by step.\n<think>"

# ----------------------#
# 1. LOAD LOCAL DATA    #
# ----------------------#

# Load from Hugging Face dataset instead of local CSV
# dataset = load_dataset("predibase/wordle-grpo", split="train")
dataset = load_dataset("predibase/wordle-grpo", split="train").to_pandas()

# Filter for valid 5-letter words only
secrets = dataset['secret'].astype(str)
total_secrets = len(secrets)
logger.info(f"Total secrets in dataset: {total_secrets}")
valid_secrets = secrets[secrets.str.len() == 5]
logger.info(f"Secrets with length 5: {len(valid_secrets)}")
# Optionally, filter for alphabetic only
valid_secrets = valid_secrets[valid_secrets.str.isalpha()]
logger.info(f"Secrets with length 5 and alphabetic only: {len(valid_secrets)}")

# Split into train/validation (80/20 split)
train_secrets, val_secrets = train_test_split(valid_secrets, test_size=0.2, random_state=42)
logger.info(f"Train set size: {len(train_secrets)}, Validation set size: {len(val_secrets)}")

# Convert to Hugging Face Datasets with 'prompt' and 'secret_word' columns
train_df = pd.DataFrame({'prompt': [render_prompt_for_dataset() for _ in train_secrets], 'secret_word': train_secrets.reset_index(drop=True)})
val_df = pd.DataFrame({'prompt': [render_prompt_for_dataset() for _ in val_secrets], 'secret_word': val_secrets.reset_index(drop=True)})
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# ----------------------#
# 3. MODEL SETUP        #
# ----------------------#

load_dotenv()
MODEL_NAME = os.getenv("HUGGINGFACE_MODEL_NAME").strip()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN").strip()

# local_rank, device = setup_distributed()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, device_map="auto")

# .to(device)

tokenizer.pad_token = tokenizer.eos_token           # reuse EOS as PAD
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id  # persist in model config
logger.info(f"Loaded model: {MODEL_NAME}")


# ----------------------#
# 3. REWARD             #
# ----------------------#


# Reward function for GRPOTrainer
def wordle_reward_func(completions, prompts=None, secret_word=None, **kwargs):
    rewards = []
    for i, completion in enumerate(completions):
        example = {
            'word_list': 'five_letter_words.csv',
            'past_guess_history': '[]',
            'secret_word': secret_word[i] if secret_word is not None else None
        }
        format_reward = output_format_check(prompts[i] if prompts else '', completion, example)
        feedback_reward = uses_previous_feedback(prompts[i] if prompts else '', completion, example)
        info_gain_reward = guess_value(prompts[i] if prompts else '', completion, example)
        reward = format_reward + feedback_reward + info_gain_reward
        rewards.append(reward)
    return rewards

# ----------------------#
# 4. MAIN               #
# ----------------------#

if __name__ == "__main__":
    logger.info("Starting GRPO training script.")
    training_args = GRPOConfig(
        output_dir="outputs/wordle-grpo",
        num_train_epochs=3,  # Number of epochs
        per_device_train_batch_size=1,  # Batch size per device
        per_device_eval_batch_size=8,   # Batch size for evaluation
        gradient_accumulation_steps=4,       # Simulates batch size of 4
        num_generations=4,       # Ensure batch size is divisible by generations
        learning_rate=1e-5,             # Example learning rate
        logging_steps=10,               # Log every 10 steps
        save_steps=100,                 # Save checkpoint every 100 steps
        eval_strategy="steps",   # Evaluate every eval_steps
        eval_steps=50,                  # Evaluate every 50 steps
        bf16=False,                     # Disable bfloat16 (A100 only)
        fp16=True,                      # Use fp16
        remove_unused_columns=False,    # Keep all columns for custom reward
        max_prompt_length=512,          # Truncate prompts if needed
        max_completion_length=4096,     # Max length for completions (updated from 32)
        seed=42,                        # Random seed
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=wordle_reward_func,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,  # ensure correct tokenization
    )
    trainer.train()
    logger.info("TRL GRPOTrainer training complete.")
