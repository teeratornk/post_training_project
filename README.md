# Wordle LLM Project

## Overview

This project simulates playing the Wordle game using a Hugging Face language model and provides a framework for evaluating model guesses using custom reward functions. It includes logging for all activities and is structured for easy extension and experimentation. The framework supports both Hugging Face datasets and local CSV word lists, and is compatible with Azure ML workflows. It now also supports RLHF/GRPO training using Hugging Face TRL, with reward functions, distributed/multi-GPU training, and Azure ML job integration.

## Features
- Loads a Hugging Face model and tokenizer using environment variables (`HUGGINGFACE_MODEL_NAME`, `HUGGINGFACE_TOKEN`).
- Defines Wordle game rules and feedback system.
- Uses a prompt template to instruct the model on how to play Wordle.
- Handles model interaction and parses the model's guesses.
- Implements the game loop, feedback mechanism, and win/loss conditions.
- Provides reward functions for evaluating guesses (format, feedback usage, information gain).
- Centralized logging for all activities, with logs saved to `outputs/reward_functions.log`.
- Supports validation and evaluation on both Hugging Face datasets and local CSV word lists.
- Outputs validation statistics to `outputs/validation_stats.json` for analysis and Azure ML compatibility.
- Modular, well-documented codebase for easy customization and extension.
- **Supports RLHF/GRPO training with Hugging Face TRL:**
  - Uses `grpo_local_data.py` for RLHF/GRPO training pipeline.
  - Integrates custom reward functions and prompt construction.
  - Compatible with distributed/multi-GPU training (torchrun/accelerate).
  - Azure ML job submission via `job.yml` and `submit_job.sh`.

## Requirements
- Python 3.7+
- `transformers` library
- `trl` (Hugging Face TRL)
- `accelerate` (for distributed/multi-GPU training)
- `python-dotenv` library
- `pandas` library
- `datasets` library (for Hugging Face datasets)
- `scikit-learn` (for data splitting)
- A Hugging Face account and access token

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up a `.env` file in the project directory with the following variables:
   ```env
   HUGGINGFACE_MODEL_NAME=your-model-name
   HUGGINGFACE_TOKEN=your-hf-access-token
   ```

## Usage
### Play a single Wordle game interactively
Run the main script directly:
```bash
python basecase_l3.py
```
By default, the script will play a game of Wordle with the secret word "BRICK".

### Validate model performance on a Hugging Face dataset
```bash
python basecase_l3_dataset.py
```
This script loads a Hugging Face dataset, filters for valid 5-letter words, splits into train/validation, and evaluates model performance on the validation set. Validation statistics are saved to `outputs/validation_stats.json`.

### Validate model performance on a local CSV word list
```bash
python basecase_l3_local_dataset.py
```
This script loads `five_letter_words.csv`, filters and splits the data, and evaluates model performance on the validation set. Statistics are saved to `outputs/validation_stats.json`.

### RLHF/GRPO Training (Distributed, Multi-GPU, Azure ML Compatible)
- Edit `grpo_local_data.py` to set your model, reward function, and training parameters.
- Submit distributed/multi-GPU jobs on Azure ML using `job.yml` and `submit_job.sh`:
  - `job.yml` is pre-configured for multi-node, multi-GPU training with either `torchrun` or `accelerate`.
  - Example command for 2 nodes × 8 GPUs each:
    ```yaml
    command: >-
      accelerate launch --num_processes 8 --main_process_port 29510 grpo_local_data.py
    resources:
      instance_count: 2
    distribution:
      type: pytorch
      process_count_per_instance: 8
    ```
  - Adjust batch size, gradient accumulation, and `num_generations` in `grpo_local_data.py` as needed to avoid batch size errors.
  - All logs and outputs are saved in the `outputs/` directory for Azure ML compatibility.

## Customization
- To change the secret word, modify the argument in the `play_game()` function at the bottom of `basecase_l3.py`.
- To use a different Hugging Face model, update the `HUGGINGFACE_MODEL_NAME` in your `.env` file.
- To adjust reward logic, edit `reward_functions.py`.
- To use a different dataset, update the dataset loading logic in `basecase_l3_dataset.py` or provide a new CSV for `basecase_l3_local_dataset.py`.
- For RLHF/GRPO, edit `grpo_local_data.py` to change model, reward, or training config.

## File Structure
- **basecase_l3.py:** Main script for simulating Wordle with a Hugging Face model.
- **basecase_l3_dataset.py:** Validates model performance on a Hugging Face dataset, logs results, and outputs statistics for Azure ML.
- **basecase_l3_local_dataset.py:** Validates model performance on a local CSV word list, logs results, and outputs statistics for Azure ML.
- **grpo_local_data.py:** RLHF/GRPO training pipeline for Wordle. This script:
  - Loads and filters the Hugging Face Wordle dataset for valid 5-letter words.
  - Splits data into train/validation sets and constructs prompts that do not leak the answer.
  - Sets up the model and tokenizer from Hugging Face Hub using environment variables.
  - Defines and applies robust, modular reward functions (output format, feedback usage, information gain) for RLHF.
  - Integrates with Hugging Face TRL's `GRPOTrainer` for RLHF/GRPO training, supporting custom reward logic.
  - Supports distributed and multi-GPU training via `torchrun` or `accelerate` (Azure ML compatible).
  - Allows configuration of batch size, gradient accumulation, number of generations, and other hyperparameters.
  - Centralizes logging and outputs all logs, checkpoints, and statistics to the `outputs/` directory for Azure ML workflows.
  - Is ready for further customization and extension for new reward functions, datasets, or training strategies.
- **reward_functions.py:** Contains reward functions for evaluating model guesses, including output format checking, use of previous feedback, and information gain.
- **logger_setup.py:** Sets up a reusable logger for the project, writing logs to `outputs/reward_functions.log`.
- **five_letter_words.csv:** Example local word list for validation.
- **job.yml:** Azure ML job configuration for distributed/multi-GPU jobs.
- **requirements.txt:** Python dependencies.
- **README.md:** Project documentation and instructions.

## Outputs
- **outputs/reward_functions.log:** Centralized log file for all activities and errors.
- **outputs/validation_stats.json:** Validation statistics for model performance, suitable for Azure ML analysis.
- **outputs/** (other): RLHF/GRPO training logs, checkpoints, and outputs.

## Example Output
```
Hugging Face model:
----------------------------------------------------------------------------------------------------
BRICK → Feedback: B(✓) R(✓) I(✓) C(✓) K(✓)
🎉 SUCCESS 🎉
```

## Azure ML Compatibility
- All logs, validation statistics, and RLHF/GRPO outputs are saved in the `outputs/` directory for easy integration with Azure ML pipelines and analysis tools.
- Distributed/multi-GPU training is supported via `torchrun` or `accelerate` in `job.yml`.

## License
This project is provided as-is for educational and research purposes.

## Acknowledgements
This project is inspired by the online course [Reinforcement Fine-Tuning LLMs with GRPO](https://learn.deeplearning.ai/courses/reinforcement-fine-tuning-llms-grpo/) from DeepLearning.AI.
