import json
import matplotlib.pyplot as plt
import os

def plot_loss(log_file, output_dir="outputs"):
    """
    Plots training and evaluation loss curves from a TRL/transformers Trainer log file.
    Args:
        log_file (str): Path to the log file (should be a JSONL file with 'loss' and 'eval_loss' keys).
        output_dir (str): Directory to save the plot.
    """
    train_steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []

    with open(log_file, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
            except Exception:
                continue
            if "loss" in record and "step" in record:
                train_steps.append(record["step"])
                train_loss.append(record["loss"])
            if "eval_loss" in record and "step" in record:
                eval_steps.append(record["step"])
                eval_loss.append(record["eval_loss"])

    plt.figure(figsize=(10, 6))
    if train_steps:
        plt.plot(train_steps, train_loss, label="Train Loss", color="blue")
    if eval_steps:
        plt.plot(eval_steps, eval_loss, label="Eval Loss", color="orange")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(plot_path)
    print(f"Loss curve saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot training and evaluation loss curves from log file.")
    parser.add_argument("--log_file", type=str, required=True, help="Path to the Trainer log file (JSONL)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save the plot")
    args = parser.parse_args()
    plot_loss(args.log_file, args.output_dir)
