#!/usr/bin/env python3
"""Extract synchronized samples for visual inspection of this run."""
import os
import argparse,json,subprocess
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps
parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
T=Path(os.environ.get('TEST_ROOT', Path(__file__).resolve().parents[1]))
paths=json.loads((T/'reports/paths.json').read_text());review=T/'reports/visual_review';review.mkdir(exist_ok=True)
frames=[0,58,150,225,325,449]
rows=[('01_dynhamr','1. Hand pose','hand_pose_overlay.mp4'),('02_targets','2. Gripper targets','retarget_overlay.mp4'),('03_arm','3. Arm masks','mask_overlay.mp4'),('04_inpaint','4. Inpainted scene','rgb/inpaint_out.mp4'),('05_robot','5. Robot IK + render','robot_overlay.mp4'),('06_composite','6. Depth composite','composite.mp4')]
fontfile='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=ImageFont.truetype(fontfile,20) if Path(fontfile).exists() else ImageFont.load_default()
small=ImageFont.truetype(fontfile,16) if Path(fontfile).exists() else ImageFont.load_default()
cellw,cellh=320,180;left,top,band=260,60,25
canvas=Image.new('RGB',(left+len(frames)*cellw+20,top+len(rows)*(cellh+band)+15),'#101820');draw=ImageDraw.Draw(canvas)
draw.text((15,15),'2026-10-19 | Cloud full-clip acceptance run | 450 frames',font=font,fill='white')
for c,i in enumerate(frames):draw.text((left+c*cellw,top-24),f'frame {i} / {i/30:.2f}s',font=small,fill='#B6C6D4')
for r,(key,label,filename) in enumerate(rows):
 if key not in paths:raise RuntimeError('Missing stage '+key)
 source=Path(paths[key])/filename;folder=review/key;folder.mkdir(exist_ok=True)
 vf='select='+ '+'.join(f'eq(n\\,{i})' for i in frames)
 if not args.render_only:
  subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(source),'-vf',vf,'-fps_mode','passthrough',str(folder/'%02d.png')],check=True)
 y=top+r*(cellh+band);draw.text((15,y+65),label,font=font,fill='white')
 for c in range(len(frames)):
  im=Image.open(folder/f'{c+1:02d}.png').convert('RGB');thumb=ImageOps.contain(im,(cellw-4,cellh-4))
  canvas.paste(thumb,(left+c*cellw+(cellw-thumb.width)//2,y+(cellh-thumb.height)//2))
canvas.save(review/'six_stages_contact_sheet.jpg',quality=92)
print(review/'six_stages_contact_sheet.jpg')
# Separate scene-depth samples keep the diagnostic visible in the report.
final=Path(paths['06_composite']);folder=review/'06_scene_depth';folder.mkdir(exist_ok=True)
if not args.render_only:
 subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(final/'scene_depth_preview.mp4'),'-vf',vf,'-fps_mode','passthrough',str(folder/'%02d.png')],check=True)
