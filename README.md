# Wordle LLM Project

## Overview

This project simulates playing the Wordle game using a Hugging Face language model and provides a framework for evaluating model guesses using custom reward functions. It includes logging for all activities and is structured for easy extension and experimentation.

## Features
- Loads a Hugging Face model and tokenizer using environment variables (`HUGGINGFACE_MODEL_NAME`, `HUGGINGFACE_TOKEN`).
- Defines Wordle game rules and feedback system.
- Uses a prompt template to instruct the model on how to play Wordle.
- Handles model interaction and parses the model's guesses.
- Implements the game loop, feedback mechanism, and win/loss conditions.
- Provides reward functions for evaluating guesses (format, feedback usage, information gain).
- Centralized logging for all activities.

## Requirements
- Python 3.7+
- `transformers` library
- `python-dotenv` library
- `pandas` library
- A Hugging Face account and access token

## Setup
1. Install dependencies:
   ```bash
   pip install transformers python-dotenv pandas
   ```
2. Set up a `.env` file in the project directory with the following variables:
   ```env
   HUGGINGFACE_MODEL_NAME=your-model-name
   HUGGINGFACE_TOKEN=your-hf-access-token
   ```

## Usage
Run the main script directly:
```bash
python basecase_l3.py
```
By default, the script will play a game of Wordle with the secret word "BRICK".

## Customization
- To change the secret word, modify the argument in the `play_game()` function at the bottom of `basecase_l3.py`.
- To use a different Hugging Face model, update the `HUGGINGFACE_MODEL_NAME` in your `.env` file.
- To adjust reward logic, edit `reward_functions.py`.

## File Structure
- **basecase_l3.py:** Main script for simulating Wordle with a Hugging Face model.
- **reward_functions.py:** Contains reward functions for evaluating model guesses, including output format checking, use of previous feedback, and information gain.
- **logger_setup.py:** Sets up a reusable logger for the project, writing logs to `reward_functions.log`.
- **README.md:** Project documentation and instructions.

## Example Output
```
Hugging Face model:
----------------------------------------------------------------------------------------------------
BRICK → Feedback: B(✓) R(✓) I(✓) C(✓) K(✓)
🎉 SUCCESS 🎉
```

## License
This project is provided as-is for educational and research purposes.

## Acknowledgements
This project is inspired by the online course [Reinforcement Fine-Tuning LLMs with GRPO](https://learn.deeplearning.ai/courses/reinforcement-fine-tuning-llms-grpo/) from DeepLearning.AI.
