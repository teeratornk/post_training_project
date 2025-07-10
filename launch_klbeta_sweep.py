import subprocess

# betas = [0.01, 0.05, 0.1, 0.2, 0.5]
betas = [0.1, 0.2, 0.5]
for beta in betas:
    print(f"Launching subprocess for KL beta={beta}")
    subprocess.run([
        "python", "grpo_local_data_sensitivity_klbeta_subprocess.py", "--beta", str(beta)
    ])
    print(f"Subprocess for KL beta={beta} completed")
