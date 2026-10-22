from pathlib import Path
import torch
from depth_anything_3.api import DepthAnything3

directory = Path(__file__).resolve().parents[1] / "assets/DA3METRIC-LARGE"
assert (directory / "model.safetensors").is_file(), "run bash prepare.sh weights"
model = DepthAnything3.from_pretrained(str(directory), local_files_only=True).eval().to("cuda")
torch.cuda.synchronize()
print("DA3METRIC-LARGE checkpoint loading passed; metric-depth inference not tested")
