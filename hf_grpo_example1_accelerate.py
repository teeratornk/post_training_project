import os
from dotenv import load_dotenv

from datasets import load_dataset, DatasetDict
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoConfig

# Load environment variables from .env file if present
load_dotenv()
MODEL_NAME = os.getenv("HUGGINGFACE_MODEL_NAME").strip()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN").strip()

def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

if __name__ == "__main__":
    # Use Huggingface split syntax for train/val split
    train_dataset = load_dataset("trl-lib/tldr", split="train[:90%]")
    val_dataset = load_dataset("trl-lib/tldr", split="train[90%:]")

    # Load model config and disable sliding window attention
    config = AutoConfig.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    if hasattr(config, 'sliding_window'):
        config.sliding_window = None
    # device_map is not set; Accelerate will handle device placement
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, config=config, token=HF_TOKEN)

    training_args = GRPOConfig(
        output_dir="outputs/Qwen2-0.5B-GRPO",
        load_best_model_at_end=True,
        greater_is_better=True,
        eval_strategy="steps",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
