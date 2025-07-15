import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# GRPO training script for Wordle using local CSV word list
# Adapted for PEFT (Parameter-Efficient Fine-Tuning)

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

# PEFT imports
from peft import get_peft_model, LoraConfig, TaskType

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
    dataset = load_dataset("predibase/wordle-grpo", split="train").to_pandas()
    valid_rows = dataset[dataset['secret'].astype(str).str.len() == 5]
    valid_rows = valid_rows[valid_rows['secret'].str.isalpha()]
    logger.info(f"Total secrets in dataset: {len(dataset)}")
    logger.info(f"Secrets with length 5 and alphabetic only: {len(valid_rows)}")
    train_rows, val_rows = train_test_split(valid_rows, test_size=0.2, random_state=42)
    logger.info(f"Train set size: {len(train_rows)}, Validation set size: {len(val_rows)}")
    train_df = train_rows[['prompt', 'secret', 'past_guess_history']].rename(columns={'secret': 'secret_word'}).reset_index(drop=True)
    val_df = val_rows[['prompt', 'secret', 'past_guess_history']].rename(columns={'secret': 'secret_word'}).reset_index(drop=True)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    inspect_data(train_df, val_df)
    return train_dataset, val_dataset

# ----------------------#
# 2. MODEL SETUP        #
# ----------------------#

def setup_model_and_tokenizer_peft():
    load_dotenv()
    MODEL_NAME = os.getenv("HUGGINGFACE_MODEL_NAME").strip()
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN").strip()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, device_map="auto")
    model.train()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Loaded model: {MODEL_NAME}")
    # PEFT config (LoRA example)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        # r=8,
        r=128,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        # target_modules=["q_proj", "v_proj"],  # Adjust based on your model architecture
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    )
    model = get_peft_model(model, peft_config)
    logger.info("Model wrapped with PEFT (LoRA)")
    return model, tokenizer

# ----------------------#
# 3. REWARD             #
# ----------------------#

def wordle_reward_func(completions, prompts=None, secret_word=None, past_guess_history=None, model=None, tokenizer=None, **kwargs):
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
    logger.info("Loading and preparing data...")
    train_dataset, val_dataset = load_and_prepare_data()
    logger.info("Data loaded and prepared.")

    logger.info("Setting up model and tokenizer with PEFT...")
    model, tokenizer = setup_model_and_tokenizer_peft()
    logger.info("Model and tokenizer setup complete.")

    def reward_func_with_model(*args, **kwargs):
        return wordle_reward_func(*args, model=model, tokenizer=tokenizer, **kwargs)
    reward_func_with_model.__name__ = "wordle_reward_func"

    logger.info("Starting GRPO training script with PEFT...")
    training_args = GRPOConfig(
        output_dir="outputs/wordle-grpo-peft",
        num_train_epochs=20,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_generations=8,
        learning_rate=1e-6,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        bf16=False,
        fp16=True,
        remove_unused_columns=False,
        max_prompt_length=1024,
        max_completion_length=2048,
        seed=42,
        gradient_checkpointing=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir="outputs/wordle-grpo-peft/logs",
        report_to=["tensorboard", "wandb"],
        run_name=f"wordle-grpo-peft-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        temperature=1.2,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1,
        generation_kwargs={
            "temperature": 1.2,
            "top_p": 0.95,
            "top_k": 40,
            "repetition_penalty": 1.1
        },
        scale_rewards=False
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func_with_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    logger.info("TRL GRPOTrainer training complete (PEFT).")

    final_model_dir = os.path.join(training_args.output_dir, "final_model")
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    logger.info(f"Final model and tokenizer saved to: {final_model_dir}")

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

    try:
        from plot_loss import plot_loss
        log_file = os.path.join(training_args.output_dir, "trainer_state.jsonl")
        if os.path.exists(log_file):
            plot_loss(log_file, output_dir=training_args.output_dir)
        else:
            logger.warning(f"Log file {log_file} not found. Skipping loss plot.")
    except Exception as e:
        logger.warning(f"Could not plot loss curves: {e}")

    logger.info(f"Trainer.state: {trainer.state}")
    for k, v in trainer.state.__dict__.items():
        logger.info(f"trainer.state.{k}: {v}")
