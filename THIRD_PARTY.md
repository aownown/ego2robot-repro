# Third-party components

This reproduction references, but does not vendor, the following work. Setup
scripts fetch official pinned revisions. The Apache-2.0 license at repository
root covers original project material only.

| Component | Role | Upstream | Terms to review |
|---|---|---|---|
| Ego2Robot | Scientific reference | https://arxiv.org/abs/2608.02580 | Citation and paper terms |
| WiLoR | Per-frame hand reconstruction | https://github.com/rolpotamias/WiLoR | CC BY-NC-ND 4.0 |
| Dyn-HaMR | Temporal hand optimization | https://github.com/ZhengdiYu/Dyn-HaMR | MIT |
| Hand-BMC-pytorch | Hand shape statistics | https://github.com/MengHao666/Hand-BMC-pytorch | MIT |
| SAM 3 | Arm segmentation | https://github.com/facebookresearch/sam3 | Meta SAM License |
| ProPainter | Inpainting | https://github.com/sczhou/ProPainter | S-Lab License 1.0, noncommercial |
| MuJoCo | Physics and rendering | https://github.com/google-deepmind/mujoco | Apache-2.0 |
| mink | Differential IK | https://github.com/kevinzakka/mink | Apache-2.0 |
| MuJoCo Menagerie | Panda model/assets | https://github.com/google-deepmind/mujoco_menagerie | Per-asset licenses |
| Depth Anything 3 | Monocular depth | https://github.com/ByteDance-Seed/Depth-Anything-3 | Apache-2.0 |
| MANO | Hand model | https://mano.is.tue.mpg.de/ | Registration and license |

See `workflow/sources.lock.json` for exact commits and
`docs/UPSTREAM_PATCHES.md` for integration changes. This table is practical
engineering guidance, not legal advice.

If you use the paper, cite:

```bibtex
@article{wang2026ego2robot,
  title={Ego2Robot: Scalable Robot Data Synthesis from Egocentric Human Data},
  author={Wang, Ye and Lin, Pei and Chen, Xiong-Hui and Yuan, Haoqi and Liang, Zhixuan and Huang, Yiyang and Chen, Anzhe and Lei, Zixing and Zhang, Jie and Zhang, Tao and Li, Haoyang and Zhang, Tong and Xiao, Chenxi and Jiao, Ziyuan and Jin, Qin},
  journal={arXiv preprint arXiv:2608.02580},
  year={2026}
}
```
