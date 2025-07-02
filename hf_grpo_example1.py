import os
from dotenv import load_dotenv


from datasets import load_dataset, DatasetDict
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoConfig

# Load environment variables from .env file if present
load_dotenv()
MODEL_NAME = os.getenv("HUGGINGFACE_MODEL_NAME").strip()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN").strip()

# Use Huggingface split syntax for train/val split
train_dataset = load_dataset("trl-lib/tldr", split="train[:90%]")
val_dataset = load_dataset("trl-lib/tldr", split="train[90%:]")

# Only train and validation splits are loaded and used. The test split is ignored.
# Huggingface Datasets may still show progress for all splits, but only train/val are used in this script.

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

# Load model config and disable sliding window attention
config = AutoConfig.from_pretrained(MODEL_NAME, token=HF_TOKEN)
if hasattr(config, 'sliding_window'):  # Some configs may use this attribute
    config.sliding_window = None
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, config=config, token=HF_TOKEN, device_map="auto")

# Update output directory for Azure ML outputs folder and enable best model selection
training_args = GRPOConfig(
    output_dir="outputs/Qwen2-0.5B-GRPO",
    load_best_model_at_end=True,
    greater_is_better=True,
    eval_strategy="steps",  # Ensure evaluation and save strategies match
)
# Update trainer to use train and eval datasets
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train and automatically select/load the best model at the end
trainer.train()