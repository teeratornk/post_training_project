# This file is adapted from the course:
# https://learn.deeplearning.ai/courses/reinforcement-fine-tuning-llms-grpo
#
# Modifications and additional logic may have been applied for this project.

import math
import re
import ast
import pandas as pd

from logger_setup import logger

# Reward/penalty constants for uses_previous_feedback
CORRECT_POSITION_REWARD = 0.2
NEW_POSITION_REWARD = 0.1
REPEATED_POSITION_PENALTY = -0.2
WRONG_LETTER_PENALTY = -0.5
EXPLORATION_REWARD = 0.05


def output_format_check(prompt: str, completion: str, example: dict) -> float:
    """
    Checks if the completion output is in the correct format and if the guess is a valid word.
    Returns a reward score based on format and validity.
    """

    reward = 0
    try:
        logger.info('Running output_format_check')
        
        # Add synthetic <think> as it's already part of the prompt and prefilled 
        # for the assistant to more easily match the regex
        completion = "<think>" + completion

        # Adjust the regex to capture only the guessed word within <guess> tags
        regex = r"<think>.*?<\/think>\s*<guess>\s*(\w+)\s*<\/guess>"

        # Search for the regex in the completion
        match = re.search(regex, completion, re.DOTALL)
        if match is None:
            logger.warning(f'output_format_check: Regex did not match. Completion: {completion}')
            return 0.0

        guess = match.group(1).strip()  # Capture the guess word
        logger.info(f"output_format_check: Guess = {guess}")

        # If the word is not 5 characters, return 0.1
        if len(guess) != 5:
            logger.info(f'output_format_check: Guess length not 5: {guess}. Completion: {completion}')
            return 0.1

        # Check if the guess is a valid word compared to a predefined list of words
        word_list = pd.read_csv(str(example["word_list"]))
        if guess not in word_list["Word"].values:
            logger.info(f'output_format_check: Guess not in word list: {guess}. Completion: {completion}')
            return 0.5

        # If everything is correct, return a reward of 1.0
        reward = 1.0
        logger.info(f'output_format_check: Success, guess={guess}, reward={reward}')
    except Exception as e:
        logger.error(f"Exception in output_format_check: {e}")
        return 0.0

    return reward

def uses_previous_feedback(prompt: str, completion: str, example: dict) -> float:
    """
    Rewards guesses that make good use of previous feedback.
    Returns a cumulative reward based on letter positions and exploration.
    """

    reward = 0
    try:
        logger.info('Running uses_previous_feedback')

        # Add synthetic <think> as it's already part of the prompt and prefilled
        completion = "<think>" + completion

        # Extract the guess from the completion
        regex = r"<guess>\s*([\w]+)\s*<\/guess>"
        match = re.search(regex, completion, re.DOTALL)
        if match is None or len(match.groups()) != 1:
            logger.warning(f'uses_previous_feedback: Regex did not match or wrong group count. Completion: {completion}')
            return 0.0

        guess = match.groups()[0].strip()
        if len(guess) != 5:
            logger.info(f'uses_previous_feedback: Guess length not 5: {guess}. Completion: {completion}')
            return 0.0

        past_guess_history = ast.literal_eval(example["past_guess_history"])
        if len(past_guess_history) == 0:
            logger.info('uses_previous_feedback: No past guesses')
            return 0.1

        correct_letter_to_position = {}
        valid_letter_to_position = {}
        wrong_letter_to_position = {}

        for past_guess, past_feedback in past_guess_history:
            past_feedback = past_feedback.split(" ")
            if len(past_feedback) != 5:
                logger.warning(f"Invalid feedback format for guess: {past_guess}")
                continue
            for i, fb in enumerate(past_feedback):
                if '✓' in fb:
                    if fb[0] not in correct_letter_to_position:
                        correct_letter_to_position[fb[0]] = set()
                    correct_letter_to_position[fb[0]].add(i)
                elif '-' in fb:
                    if fb[0] not in valid_letter_to_position:
                        valid_letter_to_position[fb[0]] = set()
                    valid_letter_to_position[fb[0]].add(i)
                else:
                    if fb[0] not in wrong_letter_to_position:
                        wrong_letter_to_position[fb[0]] = set()
                    wrong_letter_to_position[fb[0]].add(i)

        for idx, letter in enumerate(guess):
            if letter in correct_letter_to_position and idx in correct_letter_to_position[letter]:
                reward += CORRECT_POSITION_REWARD
            elif letter in valid_letter_to_position and idx not in valid_letter_to_position[letter]:
                reward += NEW_POSITION_REWARD
            elif letter in valid_letter_to_position and idx in valid_letter_to_position[letter]:
                reward += REPEATED_POSITION_PENALTY
            elif letter in wrong_letter_to_position:
                reward += WRONG_LETTER_PENALTY
            else:
                reward += EXPLORATION_REWARD

        logger.info(f'uses_previous_feedback: guess={guess}, reward={reward}')
    except Exception as e:
        logger.error(f"Exception in uses_previous_feedback: {e}")
        return 0.0

    return reward



# Reward function that computes normalized information gain of the guess, i.e.,
# does the new guess reduce the uncertainty of the secret word the most
def guess_value(prompt: str, completion: str, example: dict) -> float:
    """
    Measures how much the guess reduces uncertainty about the secret word (information gain).
    Returns the normalized expected information gain as the reward.
    """
    

    def validate_guess(secret: str, guess: str, raw_feedback: bool = False) -> str:
        feedback = []
        secret_list = list(secret)

        # Check for correct positions
        for i, (g_char, s_char) in enumerate(zip(guess, secret)):
            if g_char == s_char:
                feedback.append(f"{g_char}(✓) ")
                secret_list[i] = None
            else:
                feedback.append(None)

        # Check for misplaced letters
        for i, g_char in enumerate(guess):
            if feedback[i] is None:
                if g_char in secret_list:
                    feedback[i] = f"{g_char}(-) "
                    secret_list[secret_list.index(g_char)] = None
                else:
                    feedback[i] = f"{g_char}(x) "

        if raw_feedback:
            return feedback
        return "".join(feedback).strip()

    def filter_candidates(all_candidate_words, past_guesses):
        filtered = []
        for word in all_candidate_words:
            valid = True
            for past_guess, past_feedback in past_guesses:
                # Compute what the feedback would be if 'word' were the secret.
                candidate_feedback = validate_guess(word, past_guess)
                if candidate_feedback != past_feedback:
                    valid = False
                    break
            if valid:
                filtered.append(word)
        return filtered

    def compute_normalized_information_gain(all_candidate_words, past_guesses, guess):
        # First, filter the candidate words based on past guesses.
        candidates = filter_candidates(all_candidate_words, past_guesses)
        total_candidates = len(candidates)

        # If no candidates remain, return zeros.
        if total_candidates == 0:
            return 0.0, 0.0

        # Current uncertainty (entropy) before the guess.
        current_entropy = math.log2(total_candidates)

        # Partition candidates by the feedback pattern that would be produced by the current guess.
        feedback_groups = {}
        for word in candidates:
            # Get the raw feedback list (e.g., ['B(✓) ', 'R(✓) ', 'A(x) ', ...])
            feedback = validate_guess(word, guess, raw_feedback=True)
            # Create a simple representation for the feedback pattern.
            # '1' for correct position, '0' for wrong position, 'x' for letter not in word.
            feedback_pattern = "".join('1' if "✓" in fb else ('0' if "-" in fb else 'x') 
                                    for fb in feedback)
            feedback_groups.setdefault(feedback_pattern, []).append(word)

        expected_entropy = 0
        max_info_gain = 0
        # For each feedback group, compute its contribution to the expected entropy and the info gain.
        for group in feedback_groups.values():
            group_size = len(group)
            p = group_size / total_candidates
            # Entropy if this feedback is received.
            group_entropy = math.log2(group_size) if group_size > 0 else 0
            expected_entropy += p * group_entropy
            # Information gain for this feedback outcome.
            info_gain = current_entropy - group_entropy
            max_info_gain = max(max_info_gain, info_gain)

        # The expected gain is the reduction in entropy on average.
        expected_gain = current_entropy - expected_entropy

        # Normalize by the maximum possible gain, which is current_entropy (if you reduced to one candidate).
        normalized_expected_gain = expected_gain / current_entropy if current_entropy > 0 else 0
        normalized_max_gain = max_info_gain / current_entropy if current_entropy > 0 else 0

        return normalized_expected_gain, normalized_max_gain

    reward = 0
    try:
        logger.info('Running guess_value')
        # Add synthetic <think> as it's already part of the prompt and prefilled 
        # for the assistant to more easily match the regex
        completion = "<think>" + completion

        # Extract the guess from the completion
        # regex = r"<guess>\\s*([\\s\\S]*?)\\s*<\\/guess>$"
        regex = r"<guess>\s*(\w+)\s*<\/guess>"
        match = re.search(regex, completion, re.DOTALL)
        if match is None or len(match.groups()) != 1:
            logger.warning(f'guess_value: Regex did not match or wrong group count. Completion: {completion}')
            return 0.0

        guess = match.groups()[0].strip()
        if len(guess) != 5:
            logger.info(f'guess_value: Guess length not 5: {guess}. Completion: {completion}')
            return 0.0

        # Load the word list
        word_list = pd.read_csv(str(example["word_list"]))
        if guess not in word_list["Word"].values:
            logger.info(f'guess_value: Guess not in word list: {guess}. Completion: {completion}')
            return 0.0

        # Extract past guesses and feedback
        past_guess_history = ast.literal_eval(example["past_guess_history"])

        # Compute normalized information gain
        normalized_expected_gain, _ = compute_normalized_information_gain(
            word_list["Word"].values,
            past_guess_history,
            guess
        )

        # Compute reward based on normalized information gain
        reward = normalized_expected_gain
        logger.info(f'guess_value: guess={guess}, reward={reward}')
    except Exception as e:
        logger.error(f"Exception in guess_value: {e}")
        return 0.0

    return reward
