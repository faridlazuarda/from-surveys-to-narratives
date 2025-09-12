from safetensors.torch import load_file
import torch

# Load LoRA adapters
lora_cultural = load_file("/workspace/value-adapters/culturellm/models/cultural_context/llama/arabic/adapter_model.safetensors")
lora_normal = load_file("/workspace/value-adapters/culturellm/models/normal/llama/arabic/adapter_model.safetensors")

# Compare each tensor
for key in lora_normal.keys():
    diff = (lora_normal[key] - lora_cultural[key]).abs().sum().item()
    print(f"{key}: {diff}")
