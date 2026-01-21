# quick_gpu_test.py
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Test with chemprop
from chemprop.models import MPNN
print("Chemprop imported successfully")