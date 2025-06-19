# Wordle LLM Project

## Overview

This project simulates playing the Wordle game using a Hugging Face language model and provides a framework for evaluating model guesses using custom reward functions. It includes logging for all activities and is structured for easy extension and experimentation. The framework supports both Hugging Face datasets and local CSV word lists, and is compatible with Azure ML workflows.

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

## Requirements
- Python 3.7+
- `transformers` library
- `python-dotenv` library
- `pandas` library
- `datasets` library (for Hugging Face datasets)
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

## Customization
- To change the secret word, modify the argument in the `play_game()` function at the bottom of `basecase_l3.py`.
- To use a different Hugging Face model, update the `HUGGINGFACE_MODEL_NAME` in your `.env` file.
- To adjust reward logic, edit `reward_functions.py`.
- To use a different dataset, update the dataset loading logic in `basecase_l3_dataset.py` or provide a new CSV for `basecase_l3_local_dataset.py`.

## File Structure
- **basecase_l3.py:** Main script for simulating Wordle with a Hugging Face model.
- **basecase_l3_dataset.py:** Validates model performance on a Hugging Face dataset, logs results, and outputs statistics for Azure ML.
- **basecase_l3_local_dataset.py:** Validates model performance on a local CSV word list, logs results, and outputs statistics for Azure ML.
- **reward_functions.py:** Contains reward functions for evaluating model guesses, including output format checking, use of previous feedback, and information gain.
- **logger_setup.py:** Sets up a reusable logger for the project, writing logs to `outputs/reward_functions.log`.
- **five_letter_words.csv:** Example local word list for validation.
- **job.yml:** Azure ML job configuration.
- **requirements.txt:** Python dependencies.
- **README.md:** Project documentation and instructions.

## Outputs
- **outputs/reward_functions.log:** Centralized log file for all activities and errors.
- **outputs/validation_stats.json:** Validation statistics for model performance, suitable for Azure ML analysis.

## Example Output
```
Hugging Face model:
----------------------------------------------------------------------------------------------------
BRICK → Feedback: B(✓) R(✓) I(✓) C(✓) K(✓)
🎉 SUCCESS 🎉
```

## Azure ML Compatibility
- All logs and validation statistics are saved in the `outputs/` directory for easy integration with Azure ML pipelines and analysis tools.

## License
This project is provided as-is for educational and research purposes.

## Acknowledgements
This project is inspired by the online course [Reinforcement Fine-Tuning LLMs with GRPO](https://learn.deeplearning.ai/courses/reinforcement-fine-tuning-llms-grpo/) from DeepLearning.AI.
