import os
from pathlib import Path
import json
T=Path(os.environ.get('TEST_ROOT', Path(__file__).resolve().parents[1]))
if (T/'reports/status.json').exists():print('OVERALL:',json.loads((T/'reports/status.json').read_text())['status'])
for label in ['wilor','dynhamr','retarget','sam3','propainter','mujoco_mink','da3']:
 f=T/'reports'/('env_'+label+'.exit');print('ENV',label, f.read_text().strip() if f.exists() else 'preparing')
for p in sorted((T/'reports').glob('*.ok.json')):print('PASS',p.name.replace('.ok.json',''))
for p in sorted((T/'reports').glob('*.running.json')):
 if p.with_name(p.name.replace('.running.json','.ok.json')).exists():continue
 d=json.loads(p.read_text());last=T/'reports'/(d['stage']+'.last.json')
 failed=last.exists() and json.loads(last.read_text()).get('returncode',0)!=0
 print('FAILED' if failed else 'ACTIVE',d['stage'],d['log'])
 print('\n'.join(Path(d['log']).read_text(errors='replace').splitlines()[-3:]))
