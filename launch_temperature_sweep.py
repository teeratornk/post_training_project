import subprocess

# temperatures = [0.7, 1.0, 1.2, 1.5, 2.0]
temperatures = [1.7]
for temp in temperatures:
    print(f"Launching subprocess for temperature={temp}")
    subprocess.run([
        "python", "grpo_local_data_sensitivity_temperature_subprocess.py", "--temperature", str(temp)
    ])
    print(f"Subprocess for temperature={temp} completed")