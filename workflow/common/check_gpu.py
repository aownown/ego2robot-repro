import torch

print("torch", torch.__version__, "CUDA", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA is unavailable"
print(torch.cuda.get_device_name(), "capability", torch.cuda.get_device_capability())
x = torch.randn(128, 128, device="cuda")
assert torch.isfinite(x @ x.T).all().item()
torch.cuda.synchronize()
print("GPU matrix operation passed; this does not test model inference.")
