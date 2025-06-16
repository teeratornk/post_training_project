import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import List

from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# ----------------------#
# 1. ENV & MODEL SETUP  #
# ----------------------#

load_dotenv()
MODEL_NAME = os.getenv("HF_MODEL")
OUTPUT_DIR = "outputs"
HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# ----------------------#
# 2. PROMPT TEMPLATES   #
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
# 3. DATA STRUCTURES    #
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
# 4. PROMPT FUNCTIONS   #
# ----------------------#

def render_user_prompt(past_guesses: List[GuessWithFeedback]) -> str:
    prompt = "Make a new 5-letter word guess."
    if past_guesses:
        prompt += "\n\nHere is some previous feedback:"
        for i, guess in enumerate(past_guesses):
            prompt += f"\nGuess {i+1}: {guess}"
    return prompt

def render_prompt(past_guesses: List[GuessWithFeedback]):
    return SYSTEM_PROMPT + "\n" + render_user_prompt(past_guesses) + "\nLet me solve this step by step.\n<think>"

# ----------------------#
# 5. MODEL INTERACTION  #
# ----------------------#

def generate_stream(prompt: str) -> str:
    outputs = generator(prompt, max_new_tokens=256, temperature=0.0, do_sample=False)
    completion = outputs[0]["generated_text"][len(prompt):]
    print(completion)
    return completion

# ----------------------#
# 6. GAME LOGIC         #
# ----------------------#

def get_feedback(guess: str, secret_word: str) -> List[LetterFeedback]:
    valid_letters = set(secret_word)
    feedback = []
    for letter, secret_letter in zip(guess, secret_word):
        if letter == secret_letter:
            feedback.append(LetterFeedback.CORRECT)
        elif letter in valid_letters:
            feedback.append(LetterFeedback.WRONG_POS)
        else:
            feedback.append(LetterFeedback.WRONG_LETTER)
    return feedback

def next_turn(
    past_guesses: List[GuessWithFeedback], 
    secret_word: str
):
    prompt = render_prompt(past_guesses)
    completion = generate_stream(prompt)
    match = re.search(
        r"<guess>\s*(.*?)\s*</guess>", completion, re.DOTALL
    )
    if not match:
        raise RuntimeError("invalid guess")
    guess = match.group(1).strip().upper()
    feedback = get_feedback(guess, secret_word)
    past_guesses.append(GuessWithFeedback(guess, feedback))
    print("\n\n")
    print(("-" * 100) + "\n")
    for past_guess in past_guesses:
        print(past_guess)
    if guess == secret_word:
        print("🎉 SUCCESS 🎉")
    elif len(past_guesses) >= 6:
        print("❌ better luck next time... ❌")

# ----------------------#
# 7. EXAMPLE USAGE      #
# ----------------------#

def play_game(secret_word: str):
    past_guesses = []
    while len(past_guesses) < 6:
        try:
            next_turn(past_guesses, secret_word)
        except RuntimeError:
            print("Model did not return a valid guess. Stopping game.")
            break
        if past_guesses and past_guesses[-1].guess == secret_word:
            break

if __name__ == "__main__":
    print("Hugging Face model:")
    play_game("BRICK")