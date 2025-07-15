import os
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Starting model loading script...", flush=True)

# Hardcoded model URI
MODEL_URI = "azureml:qwen7b-grpo:1"
print("Model URI:", MODEL_URI, flush=True)
try:
    # Print contents of model URI directory
    print("Model URI contents:", os.listdir(MODEL_URI), flush=True)
    # Load your model from MODEL_URI
    model = AutoModelForCausalLM.from_pretrained(MODEL_URI)
    print("Model loaded:", type(model), flush=True)
    print("Model config:", model.config, flush=True)
    # Optional: Run a dummy inference to check model output
    tokenizer = AutoTokenizer.from_pretrained(MODEL_URI)
    inputs = tokenizer("Hello world!", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=10)
    print("Sample output:", tokenizer.decode(outputs[0], skip_special_tokens=True), flush=True)
except Exception as e:
    print("Error during model loading or inference:", e, flush=True)
