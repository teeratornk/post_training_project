import os
from dotenv import load_dotenv
# from azureml.core import Workspace, Experiment, ScriptRunConfig
# from azureml.core.compute import ComputeTarget
# from azureml.core.environment import Environment
# from azure.identity import DefaultAzureCredential
# from azureml.core.authentication import AzureCliAuthentication

from azureml.core import Workspace, Environment

ws = Workspace.from_config()  # Load your workspace from config.json
environments = Environment.list(ws)  # List all environments
print(environments)

quit()

cli_auth = AzureCliAuthentication()

# Load environment variables from .env file
load_dotenv()

# Retrieve workspace and compute details from environment variables
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
compute_target_name = os.getenv("AZURE_COMPUTE_TARGET")  # New compute target
environment_name = os.getenv("AZURE_ENVIRONMENT_NAME")  # New environment name

# Step 1: Connect to the Azure ML workspace using environment variables
ws = Workspace(
    subscription_id=subscription_id,
    resource_group=resource_group,
    workspace_name=workspace_name,
    auth=cli_auth
)

# Step 2: Define the compute target
compute_target = ComputeTarget(workspace=ws, name=compute_target_name)

# Step 3: Define the environment for the script
environment = Environment.get(workspace=ws, name=environment_name)

# Step 4: Define the script configuration
# source_directory = "./scripts"  # Path to your script directory
script_name = "test_azure_acc.py"  # Replace with your actual script name

# script_config = ScriptRunConfig(source_directory=source_directory,
#                                 script=script_name,
#                                 compute_target=compute_target,
#                                 environment=environment)
script_config = ScriptRunConfig(
    script=script_name,  # Path to the training script
    compute_target=compute_target,
    environment=environment
)

# Step 5: Create and submit the experiment
experiment_name = "accelerate-training"
experiment = Experiment(workspace=ws, name=experiment_name)

# Submit the job
run = experiment.submit(script_config)

# Step 6: Wait for completion and print the result
run.wait_for_completion(show_output=True)

# Step 7: Optionally print the run's URL to monitor it on Azure ML Studio
print(f"Run details: {run.get_portal_url()}")
