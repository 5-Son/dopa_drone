import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Test each GPU
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
# Create a tensor on GPU
if torch.cuda.is_available():
    x = torch.rand(5, 3).cuda()
    print(f"Tensor device: {x.device}")
    print("GPU computation test successful!")