# basecase_l3.py

## Overview

`basecase_l3.py` is a Python script that simulates playing the Wordle game using a Hugging Face language model. The script sets up a text-generation pipeline, defines the game logic, and interacts with the model to make guesses and process feedback according to Wordle rules.

## Features
- Loads a Hugging Face model and tokenizer using environment variables (`HUGGINGFACE_MODEL_NAME`, `HUGGINGFACE_TOKEN`).
- Defines Wordle game rules and feedback system.
- Uses a prompt template to instruct the model on how to play Wordle.
- Handles model interaction and parses the model's guesses.
- Implements the game loop, feedback mechanism, and win/loss conditions.

## Requirements
- Python 3.7+
- `transformers` library
- `python-dotenv` library
- A Hugging Face account and access token

## Setup
1. Install dependencies:
   ```bash
   pip install transformers python-dotenv
   ```
2. Set up a `.env` file in the project directory with the following variables:
   ```env
   HUGGINGFACE_MODEL_NAME=your-model-name
   HUGGINGFACE_TOKEN=your-hf-access-token
   ```

## Usage
Run the script directly:
```bash
python basecase_l3.py
```
By default, the script will play a game of Wordle with the secret word "BRICK".

## Customization
- To change the secret word, modify the argument in the `play_game()` function at the bottom of the script.
- To use a different Hugging Face model, update the `HUGGINGFACE_MODEL_NAME` in your `.env` file.

## File Structure
- **Model Setup:** Loads environment variables, model, and tokenizer.
- **Prompt Templates:** Defines the system prompt and response format for the model.
- **Data Structures:** Contains enums and dataclasses for feedback and guesses.
- **Prompt Functions:** Renders prompts for the model based on previous guesses.
- **Model Interaction:** Handles text generation and output parsing.
- **Game Logic:** Implements feedback calculation and game loop.
- **Example Usage:** Runs a sample game if the script is executed directly.

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
