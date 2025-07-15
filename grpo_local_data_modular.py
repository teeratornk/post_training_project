import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
from reward_functions_base import output_format_check, uses_previous_feedback, guess_value
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer
import torch
import datetime

# ----------------------#
# 0. UTILS              #
# ----------------------#

def inspect_data(train_df, val_df, n=3):
    """
    Print basic info and show the first n rows of train and validation DataFrames.
    """
    logger.info(f"Train DataFrame shape: {train_df.shape}")
    logger.info(f"Validation DataFrame shape: {val_df.shape}")
    print("\n--- Train DataFrame Sample ---")
    print(train_df.head(n))
    print("\n--- Validation DataFrame Sample ---")
    print(val_df.head(n))


# ----------------------#
# 1. LOAD LOCAL DATA    #
# ----------------------#

def load_and_prepare_data():
    # Load from Hugging Face dataset instead of local CSV
    dataset = load_dataset("predibase/wordle-grpo", split="train").to_pandas()

    # Filter for valid 5-letter words only
    valid_rows = dataset[dataset['secret'].astype(str).str.len() == 5]
    valid_rows = valid_rows[valid_rows['secret'].str.isalpha()]
    logger.info(f"Total secrets in dataset: {len(dataset)}")
    logger.info(f"Secrets with length 5 and alphabetic only: {len(valid_rows)}")

    # Split into train/validation (80/20 split)
    train_rows, val_rows = train_test_split(valid_rows, test_size=0.2, random_state=42)
    logger.info(f"Train set size: {len(train_rows)}, Validation set size: {len(val_rows)}")

    # Use the prompt, secret, and past_guess_history columns directly from the dataset
    train_df = train_rows[['prompt', 'secret', 'past_guess_history']].rename(columns={'secret': 'secret_word'}).reset_index(drop=True)
    val_df = val_rows[['prompt', 'secret', 'past_guess_history']].rename(columns={'secret': 'secret_word'}).reset_index(drop=True)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Inspect the data
    inspect_data(train_df, val_df)
 
    return train_dataset, val_dataset

# ----------------------#
# 2. MODEL SETUP        #
# ----------------------#

def setup_model_and_tokenizer():
    load_dotenv()
    MODEL_NAME = os.getenv("HUGGINGFACE_MODEL_NAME").strip()
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN").strip()
    # local_rank, device = setup_distributed()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, device_map="auto")
    model.train()  # Ensure model is in training mode for gradient checkpointing
    tokenizer.pad_token = tokenizer.eos_token           # reuse EOS as PAD
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id  # persist in model config
    logger.info(f"Loaded model: {MODEL_NAME}")
    return model, tokenizer

# ----------------------#
# 3. REWARD             #
# ----------------------#


# Reward function for GRPOTrainer
def wordle_reward_func(completions, prompts=None, secret_word=None, past_guess_history=None, model=None, tokenizer=None, **kwargs):
    """
    For each sample, compute reward using the prompt, secret word, and past_guess_history directly.
    Each reward function receives the raw completion for the final turn.
    """
    rewards = []
    for i in range(len(prompts)):
        base_prompt = prompts[i]
        secret = secret_word[i]
        guess_history = past_guess_history[i] if past_guess_history is not None else []
        final_completion = completions[i]
        example = {
            'word_list': 'five_letter_words.csv',
            'past_guess_history': guess_history,
            'secret_word': secret
        }
        format_reward = output_format_check(base_prompt, final_completion, example)
        feedback_reward = uses_previous_feedback(base_prompt, final_completion, example)
        info_gain_reward = guess_value(base_prompt, final_completion, example)
        episode_reward = format_reward + feedback_reward + info_gain_reward
        logger.info(f"Sample {i}: completion={final_completion}, guesses={guess_history}, "
                    f"format_reward={format_reward}, feedback_reward={feedback_reward}, info_gain_reward={info_gain_reward}, total_reward={episode_reward}")
        rewards.append(episode_reward)
    logger.info(f"Rewards for batch: {rewards}")
    return rewards


# ----------------------#
# 4. MAIN               #
# ----------------------#

if __name__ == "__main__":

    # Load the data
    logger.info("Loading and preparing data...")
    train_dataset, val_dataset = load_and_prepare_data()
    logger.info("Data loaded and prepared.")

    logger.info("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer()
    logger.info("Model and tokenizer setup complete.")

    def reward_func_with_model(*args, **kwargs):
        return wordle_reward_func(*args, model=model, tokenizer=tokenizer, **kwargs)
    reward_func_with_model.__name__ = "wordle_reward_func"

    logger.info("Starting GRPO training script.")
    
    training_args = GRPOConfig(
        output_dir="outputs/wordle-grpo",
        num_train_epochs=6,  # Number of epochs
        per_device_train_batch_size=2,  # Batch size per device
        per_device_eval_batch_size=8,   # Batch size for evaluation
        gradient_accumulation_steps=4,       # Simulates batch size of 8
        num_generations=8,       # Ensure batch size is divisible by generations
        learning_rate=1e-6,             # Example learning rate
        logging_steps=50,               # Log every 50 steps
        # save_steps=300,                 # Save checkpoint every 300 steps
        eval_strategy="steps",   # Evaluate every eval_steps
        eval_steps=50,                  # Evaluate every 50 steps
        bf16=False,                     # Disable bfloat16 (A100 only)
        fp16=True,                      # Use fp16
        remove_unused_columns=False,    # Keep all columns for custom reward
        max_prompt_length=1024,          # Truncate prompts if needed
        max_completion_length=2048,     # Max length for completions (updated from 32)
        seed=42,                        # Random seed
        gradient_checkpointing=False,
        # save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # or your custom metric
        greater_is_better=False,            # True if using reward as metric
        logging_dir="outputs/wordle-grpo/logs",
        report_to=["tensorboard", "wandb"],  # Enable logging to TensorBoard and WandB
        run_name=f"wordle-grpo-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        temperature=1.7,  # Encourage more exploration
        top_p=0.95,       # Nucleus sampling for diversity
        top_k=40,         # Top-k sampling for diversity
        repetition_penalty=1.1,  # Slight penalty to discourage repeats
        generation_kwargs={
            "temperature": 1.7,
            "top_p": 0.95,
            "top_k": 40,
            "repetition_penalty": 1.1
        },
        scale_rewards=False # Disable reward scaling for GRPO
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func_with_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,  # ensure correct tokenization
    )
    trainer.train()
    logger.info("TRL GRPOTrainer training complete.")

    # Always save the final model and tokenizer at the end, regardless of checkpoint status
    final_model_dir = os.path.join(training_args.output_dir, "final_model")
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    logger.info(f"Final model and tokenizer saved to: {final_model_dir}")

    # Find and log the best model checkpoint
    if hasattr(trainer, 'state') and hasattr(trainer.state, 'best_model_checkpoint'):
        best_ckpt = trainer.state.best_model_checkpoint
        if best_ckpt:
            logger.info(f"Best model checkpoint found at: {best_ckpt}")
            best_model_dir = os.path.join(training_args.output_dir, "best_model")
            if not os.path.exists(best_model_dir):
                os.makedirs(best_model_dir)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            logger.info(f"Best model saved to: {best_model_dir}")
        else:
            logger.warning("No best model checkpoint found.")
    else:
        logger.warning("Trainer does not have best_model_checkpoint attribute.")

    # Plot training and evaluation loss after training
    try:
        from plot_loss import plot_loss
        log_file = os.path.join(training_args.output_dir, "trainer_state.jsonl")
        if os.path.exists(log_file):
            plot_loss(log_file, output_dir=training_args.output_dir)
        else:
            logger.warning(f"Log file {log_file} not found. Skipping loss plot.")
    except Exception as e:
        logger.warning(f"Could not plot loss curves: {e}")

    # Inspect trainer.state attributes for debugging
    logger.info(f"Trainer.state: {trainer.state}")
    for k, v in trainer.state.__dict__.items():
        logger.info(f"trainer.state.{k}: {v}")
