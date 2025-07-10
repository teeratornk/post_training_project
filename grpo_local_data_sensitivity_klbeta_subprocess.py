import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import argparse
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
import wandb
import gc

def inspect_data(train_df, val_df, n=3):
    logger.info(f"Train DataFrame shape: {train_df.shape}")
    logger.info(f"Validation DataFrame shape: {val_df.shape}")
    print("\n--- Train DataFrame Sample ---")
    print(train_df.head(n))
    print("\n--- Validation DataFrame Sample ---")
    print(val_df.head(n))

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

def setup_model_and_tokenizer():
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
    return model, tokenizer

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

def main(beta):
    logger.info(f"Running training with KL beta={beta}")
    train_dataset, val_dataset = load_and_prepare_data()
    model, tokenizer = setup_model_and_tokenizer()
    def reward_func_with_model(*args, **kwargs):
        return wordle_reward_func(*args, model=model, tokenizer=tokenizer, **kwargs)
    reward_func_with_model.__name__ = "wordle_reward_func"
    training_args = GRPOConfig(
        output_dir=f"outputs/wordle-grpo-klbeta{beta}",
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        num_generations=8,
        learning_rate=1e-6,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=50,
        bf16=False,
        fp16=True,
        remove_unused_columns=False,
        max_prompt_length=1024,
        max_completion_length=2048,
        seed=42,
        gradient_checkpointing=False,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=f"outputs/wordle-grpo-klbeta{beta}/logs",
        report_to=["tensorboard", "wandb"],
        run_name=f"wordle-grpo-klbeta{beta}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        temperature=1.2,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1,
        beta=beta,
        generation_kwargs={
            "temperature": 1.2,
            "top_p": 0.95,
            "top_k": 40,
            "repetition_penalty": 1.1
        },
        use_liger_loss=True
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
    logger.info(f"TRL GRPOTrainer training complete for KL beta={beta}.")
    try:
        wandb.finish()
    except ImportError:
        logger.warning("wandb not installed; skipping wandb.finish().")
    del trainer
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    try:
        from plot_loss import plot_loss
        log_file = os.path.join(training_args.output_dir, "trainer_state.jsonl")
        if os.path.exists(log_file):
            plot_loss(log_file, output_dir=training_args.output_dir)
        else:
            logger.warning(f"Log file {log_file} not found. Skipping loss plot.")
    except Exception as e:
        logger.warning(f"Could not plot loss curves for KL beta={beta}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, required=True, help="KL beta value for GRPOConfig")
    args = parser.parse_args()
    main(args.beta)
