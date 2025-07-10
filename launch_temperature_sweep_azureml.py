# launch_temperature_sweep_azureml.py
import os
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, ComputeTarget
from azureml.core.runconfig import RunConfiguration

# List of temperatures to sweep
temperatures = [0.7, 1.0, 1.2, 1.5, 2.0]

# Azure ML workspace config (assumes config.json is present)
ws = Workspace.from_config()
experiment = Experiment(ws, "grpo_sen_temperature_multi_node")

# Use the same environment and compute as in job.yml
env = Environment.get(ws, name="tkad15-grpo")
compute_target = ComputeTarget(workspace=ws, name="tkad15-8-v100-westus2")

for temp in temperatures:
    run_config = RunConfiguration()
    run_config.environment = env
    script_args = ["--temperature", str(temp)]
    src = ScriptRunConfig(
        source_directory=".",
        script="grpo_local_data_sensitivity_temperature_subprocess.py",
        arguments=script_args,
        compute_target=compute_target,
        run_config=run_config
    )
    run = experiment.submit(src)
    print(f"Submitted job for temperature={temp}, run id: {run.id}")
