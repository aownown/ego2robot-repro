#!/usr/bin/env python3
import os
from pathlib import Path
import concurrent.futures,fractions,json,subprocess
T=Path(os.environ.get('TEST_ROOT', Path(__file__).resolve().parents[1]));paths=json.loads((T/'reports/paths.json').read_text())
files=[]
for key,folder in paths.items():
 if key=='06_composite':continue  # All four final streams are independently checked by stage 6.
 files.extend((key,p) for p in Path(folder).rglob('*.mp4') if '.partial.' not in p.name)
def check(item):
 key,p=item
 d=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,nb_read_frames,avg_frame_rate','-of','json',str(p)],text=True))['streams'][0]
 assert int(d['nb_read_frames'])==450,(p,d)
 assert fractions.Fraction(d['avg_frame_rate'])==30,(p,d)
 return dict(stage=key,path=str(p),**d,status='passed')
results=list(concurrent.futures.ThreadPoolExecutor(max_workers=4).map(check,files))
assert results
(T/'reports/intermediate_videos.json').write_text(json.dumps({'status':'passed','videos_verified':len(results),'expected_frames':450,'expected_fps':30,'videos':results},indent=2))
print('Verified',len(results),'intermediate video streams, each 450 frames at 30 fps')
