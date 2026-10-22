from pathlib import Path
import torch
from sam3.model_builder import build_sam3_video_model

weight = Path(__file__).resolve().parents[1] / "assets/sam3/sam3.pt"
assert weight.is_file(), "run bash prepare.sh weights"
model = build_sam3_video_model(checkpoint_path=str(weight), load_from_HF=False, compile=False)
model.eval().to("cuda")
torch.cuda.synchronize()
print("SAM 3 video model checkpoint loading passed; propagation not tested")
