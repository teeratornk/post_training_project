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
  - Uses `grpo_local_data.py` and `hf_grpo_example1_accelerate.py`/`hf_grpo_example1_ddp.py` for RLHF/GRPO training pipeline.
  - Integrates custom reward functions and prompt construction.
  - Compatible with distributed/multi-GPU training (torchrun/accelerate/AzureML DDP).
  - Azure ML job submission via `job.yml` and `submit_job.sh`.
  - Supports logging to TensorBoard and Weights & Biases (wandb) for training monitoring.

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
- Edit `grpo_local_data.py`, `hf_grpo_example1_accelerate.py`, or `hf_grpo_example1_ddp.py` to set your model, reward function, and training parameters.
- Submit distributed/multi-GPU jobs on Azure ML using `job.yml` and `submit_job.sh`:
  - `job.yml` is pre-configured for multi-node, multi-GPU training with either `torchrun` or `accelerate`.
  - Example command for 2 nodes × 8 GPUs each (Accelerate):
    ```yaml
    command: >-
      accelerate launch --config_file acc_config.yaml grpo_local_data.py
    resources:
      instance_count: 2
    ```
  - To use the DDP script:
    ```yaml
    command: >-
      python hf_grpo_example1_ddp.py
    resources:
      instance_count: 2
    distribution:
      type: pytorch
      process_count_per_instance: 8
    ```
  - To use the Accelerate script:
    ```yaml
    command: >-
      accelerate launch --config_file acc_config.yaml hf_grpo_example1_accelerate.py
    resources:
      instance_count: 2
    ```
  - Adjust batch size, gradient accumulation, and `num_generations` in your script as needed to avoid batch size errors.
  - All logs and outputs are saved in the `outputs/` directory for Azure ML compatibility.
  - For wandb logging, set the `WANDB_API_KEY` and (optionally) `WANDB_PROJECT` environment variables in your AzureML job/environment.

### Launch Temperature Sweep
- **launch_temperature_sweep.py:** Launches a temperature sweep by running each temperature value sequentially as a subprocess on the same Azure ML node. Each run is executed one after another, sharing the same resources.
- **grpo_local_data_sensitivity_temperature_subprocess.py:** Modified version of the sensitivity analysis script that accepts a `--temperature` argument and runs a single temperature value per process. Designed for use with subprocess-based or Azure ML job-based sweeps.
- **launch_klbeta_sweep.py:** Launches a KL beta sweep by running each beta value sequentially as a subprocess. Each run is executed one after another, sharing the same resources. Edit the `betas` list in this script to set the desired beta values.
- **grpo_local_data_sensitivity_klbeta_subprocess.py:** Modified version of the sensitivity analysis script that accepts a `--beta` argument and runs a single KL beta value per process. This script sets the `beta` parameter in `GRPOConfig` to include the KL divergence term, enabling experiments on the effect of KL regularization.
- **launch_temperature_sweep_azureml.py:** Submits a separate Azure ML job for each temperature value using the Azure ML Python SDK. Each job runs `grpo_local_data_sensitivity_temperature_subprocess.py` with a different `--temperature` argument, allowing jobs to be distributed across multiple nodes/VMs in your Azure ML compute cluster. This is the recommended approach for running each temperature on a separate node in parallel.

#### How to use `launch_temperature_sweep_azureml.py`
1. Ensure you have a valid `config.json` for your Azure ML workspace in your project directory.
2. Make sure your Azure ML environment (`tkad15-grpo`) and compute target (`tkad15-8-v100-westus2`) exist.
3. Install the Azure ML Python SDK if needed: `pip install azureml-core`.
4. Run the script:
   ```pwsh
   python launch_temperature_sweep_azureml.py
   ```
5. Each temperature value will be submitted as a separate job to Azure ML, and jobs will be distributed across available nodes in your compute cluster. Monitor progress in Azure ML Studio.

#### How to use KL Beta Sweep
1. Edit `launch_klbeta_sweep.py` to set the desired KL beta values in the `betas` list.
2. Run the script:
   ```pwsh
   python launch_klbeta_sweep.py
   ```
3. Each beta value will be run as a separate subprocess, executing `grpo_local_data_sensitivity_klbeta_subprocess.py` with the corresponding `--beta` argument. Outputs and logs are saved in separate directories for each beta value.

#### How to use `grpo_local_data_sensitivity_klbeta_subprocess.py` manually
To run a single KL beta experiment:
```pwsh
python grpo_local_data_sensitivity_klbeta_subprocess.py --beta 0.1
```

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
- **grpo_local_data_modular.py:** Modular RLHF/GRPO training script for Wordle. This script:
  - Loads and filters the Hugging Face Wordle dataset for valid 5-letter words, splits into train/validation sets, and constructs prompts.
  - Sets up the model and tokenizer from Hugging Face Hub using environment variables.
  - Defines modular reward functions (output format, feedback usage, information gain) for RLHF.
  - Integrates with Hugging Face TRL's `GRPOTrainer` for RLHF/GRPO training, supporting custom reward logic.
  - Allows configuration of batch size, gradient accumulation, number of generations, temperature, and other hyperparameters.
  - Centralizes logging and outputs all logs, checkpoints, and statistics to the `outputs/` directory.
  - Plots and saves training/evaluation loss curves after training.
  - Designed for easy extension and experimentation with new reward functions, datasets, or training strategies.
- **grpo_local_data_sensitivity_temperature.py:** Script for sensitivity analysis of the temperature parameter in GRPO training. This script:
  - Loops over a range of temperature values (e.g., 0.7, 1.0, 1.2, 1.5, 2.0) and runs a full training session for each.
  - For each temperature, re-initializes the model and tokenizer to ensure independent runs.
  - Explicitly deletes model, tokenizer, and trainer objects and calls garbage collection and CUDA memory cleanup after each run to prevent GPU memory leaks.
  - Calls `wandb.finish()` after each run to ensure separate Weights & Biases runs for each temperature.
  - Plots and saves training/evaluation loss curves for each temperature in separate output directories.
  - Useful for analyzing the effect of temperature on model diversity and performance in RLHF/GRPO training.
- **launch_temperature_sweep.py:** Launches a temperature sweep by running each temperature value sequentially as a subprocess on the same Azure ML node. Each run is executed one after another, sharing the same resources.
- **grpo_local_data_sensitivity_temperature_subprocess.py:** Modified version of the sensitivity analysis script that accepts a `--temperature` argument and runs a single temperature value per process. Designed for use with subprocess-based or Azure ML job-based sweeps.
- **launch_klbeta_sweep.py:** Launches a KL beta sweep by running each beta value sequentially as a subprocess. Each run is executed one after another, sharing the same resources. Edit the `betas` list in this script to set the desired beta values.
- **grpo_local_data_sensitivity_klbeta_subprocess.py:** Modified version of the sensitivity analysis script that accepts a `--beta` argument and runs a single KL beta value per process. This script sets the `beta` parameter in `GRPOConfig` to include the KL divergence term, enabling experiments on the effect of KL regularization.
- **launch_temperature_sweep_azureml.py:** Submits a separate Azure ML job for each temperature value using the Azure ML Python SDK. Each job runs `grpo_local_data_sensitivity_temperature_subprocess.py` with a different `--temperature` argument, allowing jobs to be distributed across multiple nodes/VMs in your Azure ML compute cluster. This is the recommended approach for running each temperature on a separate node in parallel.
- **reward_functions.py:** Contains reward functions for evaluating model guesses, including output format checking, use of previous feedback, and information gain.
- **logger_setup.py:** Sets up a reusable logger for the project, writing logs to `outputs/reward_functions.log`.
- **five_letter_words.csv:** Example local word list for validation.
- **job.yml:** Azure ML job configuration for distributed/multi-GPU jobs.
- **requirements.txt:** Python dependencies.
- **README.md:** Project documentation and instructions.
- **hf_grpo_example1.py:** Example script to demonstrate GRPO fine-tuning workflow.
- **hf_grpo_example1_ddp.py:** DDP-compatible GRPO training script for AzureML PyTorch distributed jobs. Logs to TensorBoard and wandb.
- **hf_grpo_example1_accelerate.py:** Accelerate-compatible GRPO training script for multi-GPU/Accelerate jobs. Logs to TensorBoard and wandb.

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

## hf_grpo_example1.py, hf_grpo_example1_ddp.py, and hf_grpo_example1_accelerate.py

### hf_grpo_example1.py
This script demonstrates how to fine-tune a Huggingface causal language model using GRPO (Generalized Reinforcement Policy Optimization) on the TLDR dataset. It loads environment variables from a `.env` file for Huggingface credentials and model selection, splits the dataset into training and validation sets using Huggingface's built-in split syntax, and defines a reward function that encourages completions close to 20 characters. The script disables sliding window attention if present, tracks the best model based on validation reward, and saves the best checkpoint in the `outputs` folder for Azure ML compatibility.

### hf_grpo_example1_ddp.py
DDP-compatible GRPO training script for AzureML PyTorch distributed jobs. Logs to TensorBoard and wandb. Use this script with AzureML's `distribution` block for multi-GPU training.

### hf_grpo_example1_accelerate.py
Accelerate-compatible GRPO training script for multi-GPU/Accelerate jobs. Logs to TensorBoard and wandb. Use this script with `accelerate launch` and optionally a config file (e.g., `acc_config.yaml`).
