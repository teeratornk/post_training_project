#!/bin/bash

# Define the name of the new environment
ENV_NAME="azure_ml_env"

# Create the new conda environment
echo "Creating a new conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

# Activate the new environment
echo "Activating the $ENV_NAME environment"
conda activate $ENV_NAME

# # Install Azure ML SDK
echo "Installing Azure ML SDK"
pip install azureml-sdk  # This installs both azureml-core and other components

# # Confirm the environment setup
echo "Installation complete! Your environment '$ENV_NAME' is ready to submit jobs to Azure ML Studio."
