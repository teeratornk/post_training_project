#!/bin/bash

# Usage:
# ./submit_job.sh <workspace-name> <resource-group> <subscription-id>

# Input variables
WORKSPACE_NAME=$1
RESOURCE_GROUP=$2
SUBSCRIPTION_ID=$3

# Echo inputs
echo "📦 Workspace Name: $WORKSPACE_NAME"
echo "🧾 Resource Group: $RESOURCE_GROUP"
echo "📄 Subscription ID: $SUBSCRIPTION_ID"

# Check .env
if [ ! -f .env ]; then
  echo "❌ .env file not found. Exiting."
  exit 1
fi

# Load environment variables from .env
set -o allexport
source .env
set +o allexport

# Check if job.yml exists
if [ ! -f job.yml ]; then
  echo "❌ job.yml not found in the azureml directory. Exiting."
  exit 1
fi

# Create temporary job file with env vars injected
TEMP_JOB_FILE="job_temp.yml"
cp job.yml "$TEMP_JOB_FILE"

echo "🛠️ Injecting environment variables into $TEMP_JOB_FILE"

cat <<EOL >> "$TEMP_JOB_FILE"

environment_variables:
  AZURE_OPENAI_API_KEY: "$AZURE_OPENAI_API_KEY"
  AZURE_OPENAI_ENDPOINT: "$AZURE_OPENAI_ENDPOINT"
  AZURE_OPENAI_MODEL: "$AZURE_OPENAI_MODEL"
  AZURE_OPENAI_API_VERSION: "$AZURE_OPENAI_API_VERSION"
  HUGGINGFACE_TOKEN: "$HUGGINGFACE_TOKEN"
  HUGGINGFACE_MODEL_NAME: "$HUGGINGFACE_MODEL_NAME"
  HUGGINGFACE_MODEL_PATH: "$HUGGINGFACE_MODEL_PATH"
EOL

# Submit the job
echo "🚀 Submitting Azure ML job..."
az ml job create \
  --file "$TEMP_JOB_FILE" \
  --workspace-name "$WORKSPACE_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --subscription "$SUBSCRIPTION_ID"

# Clean up
rm "$TEMP_JOB_FILE"

echo "✅ Job submission completed."