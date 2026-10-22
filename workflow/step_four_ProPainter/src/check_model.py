import sys
from pathlib import Path
import torch

repo = Path(__file__).resolve().parents[1] / "repos/ProPainter"
sys.path.insert(0, str(repo))
from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator

raft = RAFT_bi(str(repo / "weights/raft-things.pth"), "cuda")
flow = RecurrentFlowCompleteNet(str(repo / "weights/recurrent_flow_completion.pth")).eval().cuda()
model = InpaintGenerator(model_path=str(repo / "weights/ProPainter.pth")).eval().cuda()
torch.cuda.synchronize()
print("ProPainter, recurrent flow and RAFT weights loaded; video inference not tested")
