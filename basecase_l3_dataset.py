import os
import re
import json
from dataclasses import dataclass
from enum import Enum
from typing import List

from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from logger_setup import logger
from sklearn.model_selection import train_test_split

# ----------------------#
# 1. ENV & MODEL SETUP  #
# ----------------------#

load_dotenv()
MODEL_NAME = os.getenv("HUGGINGFACE_MODEL_NAME").strip()
OUTPUT_DIR = "outputs"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN").strip()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)

tokenizer.pad_token = tokenizer.eos_token           # reuse EOS as PAD
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id  # persist in model config

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
logger.info(f"Loaded model: {MODEL_NAME}")

# ----------------------#
# 2. DATASET LOADING    #
# ----------------------#

dataset = load_dataset("predibase/wordle-grpo", split="train")
dataset = dataset.to_pandas()

# Inspect dataset columns and a sample row
print('Dataset columns:', dataset.columns.tolist())
print('First 3 rows:')
print(dataset.head(3))
logger.info(f"Dataset columns: {dataset.columns.tolist()}")
logger.info(f"First row: {dataset.head(1).to_dict()}")

# ----------------------#
# 3. PROMPT TEMPLATES   #
# ----------------------#

SYSTEM_PROMPT = """
You are playing Wordle, a word-guessing game.

### Game Rules:
- You have **6 tries** to guess a secret **5-letter** word.
- Each guess must be a valid **5-letter English word**.
- After each guess, you will receive feedback indicating how close 
your guess was.

### Feedback Format:
Each letter in your guess will receive one of three symbols:
1. ✓ : The letter is in the word and in the CORRECT position.
2. - : The letter is in the word but in the WRONG position.
3. x : The letter is NOT in the word.

### Example:
Secret Word: BRISK

Guess 1: STORM → Feedback: S(-) T(x) O(x) R(-) M(x)
Guess 2: BRAVE → Feedback: B(✓) R(✓) A(x) V(x) E(x)
Guess 3: BRISK → Feedback: B(✓) R(✓) I(✓) S(✓) K(✓)

### Response Format:
Think through the problem and feedback step by step. Make sure to 
first add your step by step thought process within <think> </think> 
tags. Then, return your guessed word in the following format: 
<guess> guessed-word </guess>.
"""

# ----------------------#
# 4. DATA STRUCTURES    #
# ----------------------#

class LetterFeedback(Enum):
    CORRECT = "✓"
    WRONG_POS = "-"
    WRONG_LETTER = "x"

@dataclass
class GuessWithFeedback:
    guess: str
    feedback: List[LetterFeedback]

    def __repr__(self) -> str:
        feedback_str = " ".join(f"{letter}({fb.value})" for letter, fb in zip(self.guess, self.feedback))
        return f"{self.guess} → Feedback: {feedback_str}"

# ----------------------#
# 5. PROMPT FUNCTIONS   #
# ----------------------#

def render_user_prompt(past_guesses: List[GuessWithFeedback]) -> str:
    logger.info(f"Rendering user prompt with {len(past_guesses)} past guesses.")
    prompt = "Make a new 5-letter word guess."
    if past_guesses:
        prompt += "\n\nHere is some previous feedback:"
        for i, guess in enumerate(past_guesses):
            prompt += f"\nGuess {i+1}: {guess}"
    return prompt

def render_prompt(past_guesses: List[GuessWithFeedback]):
    return SYSTEM_PROMPT + "\n" + render_user_prompt(past_guesses) + "\nLet me solve this step by step.\n<think>"

# ----------------------#
# 6. MODEL INTERACTION  #
# ----------------------#

def generate_stream(prompt: str) -> str:
    logger.info("Generating model output for prompt.")
    outputs = generator(prompt, max_new_tokens=256, do_sample=False)
    completion = outputs[0]["generated_text"][len(prompt):]
    logger.info(f"Model completion: {completion.strip()[:100]}")
    print(completion)
    return completion

# ----------------------#
# 7. GAME LOGIC         #
# ----------------------#

def get_feedback(guess: str, secret_word: str) -> List[LetterFeedback]:
    logger.info(f"Calculating feedback for guess '{guess}' against secret '{secret_word}'")
    valid_letters = set(secret_word)
    feedback = []
    for letter, secret_letter in zip(guess, secret_word):
        if letter == secret_letter:
            feedback.append(LetterFeedback.CORRECT)
        elif letter in valid_letters:
            feedback.append(LetterFeedback.WRONG_POS)
        else:
            feedback.append(LetterFeedback.WRONG_LETTER)
    logger.info(f"Feedback: {[fb.value for fb in feedback]}")
    return feedback

def next_turn(
    past_guesses: List[GuessWithFeedback], 
    secret_word: str
):
    logger.info(f"Starting next turn. Past guesses: {len(past_guesses)}")
    prompt = render_prompt(past_guesses)
    completion = generate_stream(prompt)
    match = re.search(
        r"<guess>\s*(.*?)\s*</guess>", completion, re.DOTALL
    )
    if not match:
        logger.warning("Model did not return a valid guess. Skipping this turn.")
        logger.info(f"Invalid guess output: {completion.strip()[:200]}")
        print("Warning: Model did not return a valid guess. Skipping this turn.")
        return False  # Indicate invalid guess, but do not raise
    guess = match.group(1).strip().upper()
    logger.info(f"Model guessed: {guess}")
    feedback = get_feedback(guess, secret_word)
    past_guesses.append(GuessWithFeedback(guess, feedback))
    print("\n\n")
    print(("-" * 100) + "\n")
    for past_guess in past_guesses:
        print(past_guess)
    if guess == secret_word:
        logger.info("Game won!")
        print("🎉 SUCCESS 🎉")
    elif len(past_guesses) >= 6:
        logger.info("Game lost: max turns reached.")
        print("❌ better luck next time... ❌")
    return True  # Indicate valid guess

# ----------------------#
# 8. EXAMPLE USAGE      #
# ----------------------#

# Filter for valid 5-letter words only
secrets = dataset['secret'].astype(str)
total_secrets = len(secrets)
logger.info(f"Total secrets in dataset: {total_secrets}")
valid_secrets = secrets[secrets.str.len() == 5]
logger.info(f"Secrets with length 5: {len(valid_secrets)}")
# Optionally, filter for alphabetic only
valid_secrets = valid_secrets[valid_secrets.str.isalpha()]
logger.info(f"Secrets with length 5 and alphabetic only: {len(valid_secrets)}")

# Split into train/validation (80/20 split)
train_secrets, val_secrets = train_test_split(valid_secrets, test_size=0.2, random_state=42)
logger.info(f"Train set size: {len(train_secrets)}, Validation set size: {len(val_secrets)}")

def play_game_on_validation(val_secrets):
    logger.info(f"Validating base model on {len(val_secrets)} secret words.")
    total = len(val_secrets)
    success = 0
    guess_counts = []
    stats = []
    for idx, secret_word in enumerate(val_secrets):
        print(f"\n===== GAME {idx+1} (Secret: {secret_word}) =====")
        past_guesses = []
        turn = 1
        win = False
        while turn <= 6:
            print(f"\nTurn {turn}:")
            valid_guess = next_turn(past_guesses, secret_word)
            if not valid_guess:
                logger.warning("No valid guess this turn.")
                print("No valid guess this turn.")
            if past_guesses and past_guesses[-1].guess == secret_word:
                win = True
                break
            turn += 1
        # Summary
        logger.info(f"Game {idx+1} ended. Win: {win}. Turns: {len(past_guesses)}")
        print("\n===== GAME SUMMARY =====")
        print(f"Secret word: {secret_word}")
        print(f"Result: {'WIN' if win else 'LOSS'} in {len(past_guesses)} turn(s)")
        print("Guesses:")
        for i, guess in enumerate(past_guesses, 1):
            print(f"  {i}: {guess}")
        stats.append({
            'secret_word': secret_word,
            'win': win,
            'num_guesses': len(past_guesses) if win else None
        })
        if win:
            success += 1
            guess_counts.append(len(past_guesses))
    print(f"\nValidation complete. Success: {success}/{total}")
    if guess_counts:
        avg_guesses = sum(guess_counts) / len(guess_counts)
        print(f"Average guesses (for successful games): {avg_guesses:.2f}")
    else:
        print("No successful games.")
    # Save statistics to outputs folder
    os.makedirs('outputs', exist_ok=True)
    stats_summary = {
        'total': total,
        'success': success,
        'fail': total - success,
        'avg_guesses_success': sum(guess_counts) / len(guess_counts) if guess_counts else None,
        'details': stats
    }
    with open(os.path.join('outputs', 'validation_stats.json'), 'w') as f:
        json.dump(stats_summary, f, indent=2)
    logger.info(f"Saved validation statistics to outputs/validation_stats.json")

if __name__ == "__main__":
    logger.info("Hugging Face model:")
    play_game_on_validation(val_secrets)
