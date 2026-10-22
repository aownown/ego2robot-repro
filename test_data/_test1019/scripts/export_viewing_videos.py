import os
from pathlib import Path
import subprocess,json
T=Path(os.environ.get('TEST_ROOT', Path(__file__).resolve().parents[1]));W=T.parents[1]/'workflow';p=json.loads((T/'reports/paths.json').read_text())
argv=[str(W/'step_two_Retarget_and_Smooth/envs/retarget/bin/python'),str(W/'common/export_missing_videos.py'),'--clip',str(T.parent),'--step-one',p['01_dynhamr'],'--step-two',p['02_targets'],'--arm-run',p['03_arm'],'--shadow-run',p['03_shadow'],'--prepared-run',p['03_prepared'],'--background',p['04_inpaint'],'--robot',p['05_robot'],'--report',str(T/'reports/video_exports.json')]
(T/'reports/viewing_video_export_command.json').write_text(json.dumps(argv,indent=2))
subprocess.run(argv,check=True)
