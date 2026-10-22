#!/usr/bin/env python3
import os
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess
T=Path(os.environ.get('TEST_ROOT', Path(__file__).resolve().parents[1]));W=T.parents[1]/'workflow';dest=T/'reports/provenance';dest.mkdir(exist_ok=True)
versions={};code={}
for stage in sorted(W.glob('step_*')):
 if not stage.is_dir():continue
 files=list(stage.glob('*.sh'))+list(stage.glob('requirements*.txt'))+list((stage/'configs').glob('*'))+list((stage/'src').glob('*.py'))
 code.update({str(p.relative_to(W)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()})
 for repo in (stage/'repos').iterdir():
  if not (repo/'.git').exists():continue
  key=stage.name+'/'+repo.name
  def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True)
  versions[key]={'revision':git('rev-parse','HEAD').strip(),'status':git('status','--porcelain')}
  patch=git('diff','--no-ext-diff','HEAD')
  if patch:(dest/(stage.name+'__'+repo.name+'.diff')).write_text(patch)
 for lock in (stage/'logs').glob('*pip-freeze.txt'):shutil.copy2(lock,dest/(stage.name+'__'+lock.name))
 for lock in (stage/'logs').glob('*conda-explicit.txt'):shutil.copy2(lock,dest/(stage.name+'__'+lock.name))
code.update({str(p.relative_to(W)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (W/'common').glob('*') if p.is_file() and p.suffix in ('.py','.sh')})
code.update({str(p.relative_to(T)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (T/'scripts').glob('*') if p.is_file()})
result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repos':versions,'source_sha256':code}
(dest/'code_versions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(dest)
