import torch
import sys

def check():
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device Index: {torch.cuda.current_device()}")
    else:
        print("CUDA NOT AVAILABLE - System is using CPU!")
    
if __name__ == "__main__":
    check()
