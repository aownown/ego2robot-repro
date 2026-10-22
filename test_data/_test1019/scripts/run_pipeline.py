#!/usr/bin/env python3
"""Execute this server acceptance run. Keep every failed attempt and its log."""
import argparse, datetime, fcntl, hashlib, json, os, shlex, subprocess, time
from pathlib import Path
P=Path(os.environ.get('PROJECT', Path(__file__).resolve().parents[3])); W=P/'workflow'; T=P/'test_data/_test1019'; C=P/'test_data'
S={1:W/'step_one_WiLoR_and_DynHaMR',2:W/'step_two_Retarget_and_Smooth',3:W/'step_three_SAM3',4:W/'step_four_ProPainter',5:W/'step_five_MuJoCo_and_mink',6:W/'step_six_DepthAnything3_and_Composite'}
ENV={1:'wilor',2:'retarget',3:'sam3',4:'propainter',5:'mujoco_mink',6:'da3'}
def py(s,env=None):return str(S[s]/'envs'/(env or ENV[s])/'bin/python')
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p,default):return json.loads(p.read_text()) if p.exists() else default

def wait_env(name):
 f=T/'reports'/('env_'+name+'.exit')
 while not f.exists():
  print(now(),'waiting for environment',name,flush=True);time.sleep(15)
 if f.read_text().strip()!='0':raise RuntimeError('Environment setup failed: '+name)

def paths():return read(T/'reports/paths.json',{})
def out(k):return paths()[k]
def execute(label,envs,builder,key=None,checks=False):
 with (T/'reports'/(label+'.lock')).open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for name in envs:wait_env(name)
  mark=T/'reports'/(label+'.ok.json')
  previous=read(mark,{})
  # Builders use the current dependency paths, so changed inputs invalidate a marker.
  if previous:
   candidate=builder(previous.get('output'))
   if previous.get('argv')==candidate:
    print(now(),'verified completed marker',label,flush=True);return
  target=None
  if key:
   base=T/'outputs'/key;target=base;i=1
   while target.exists():i+=1;target=base.with_name(base.name+'_attempt'+str(i))
   with (T/'reports/paths.lock').open('w') as plock:
    fcntl.flock(plock,fcntl.LOCK_EX);mapping=paths();mapping[key]=str(target)
    (T/'reports/paths.json').write_text(json.dumps(mapping,indent=2))
  argv=builder(str(target) if target else None)
  stamp=now(); log=T/'logs'/(label+'_'+datetime.datetime.now().strftime('%H%M%S')+'.log')
  record={'stage':label,'started':stamp,'argv':argv,'output':str(target) if target else None,'log':str(log)}
  (T/'reports'/(label+'.running.json')).write_text(json.dumps(record,indent=2))
  print(stamp,'START',label,shlex.join(argv),'LOG='+str(log),flush=True)
  e=os.environ.copy();e['CUDA_VISIBLE_DEVICES']='3' if label in ('03_arm','03_shadow') else '2'
  with log.open('w') as f:
   f.write(shlex.join(argv)+'\n');f.flush()
   code=subprocess.call(argv,env=e,cwd=str(W),stdout=f,stderr=subprocess.STDOUT)
  record.update(finished=now(),returncode=code)
  (T/'reports'/(label+'.last.json')).write_text(json.dumps(record,indent=2))
  if code:raise RuntimeError(label+' failed, see '+str(log))
  mark.write_text(json.dumps(record,indent=2)); print(now(),'DONE',label,flush=True)

JOBS=[
 ('01_wilor',['wilor'],'01_wilor',lambda o:[py(1),'-u',str(S[1]/'src/export_wilor.py'),'--clip',str(C),'--output',o]),
 ('01_hand_masks',['sam3'],'01_hand_masks',lambda o:[py(3),'-u',str(S[1]/'src/sam3_hand_masks.py'),'--input',out('01_wilor'),'--output',o]),
 ('01_track',['wilor'],'01_track',lambda o:[py(1),'-u',str(S[1]/'src/track_wilor.py'),'--input',out('01_wilor'),'--hand-masks',out('01_hand_masks'),'--output',o]),
 ('01_dynhamr',['dynhamr'],'01_dynhamr',lambda o:[py(1,'dynhamr'),'-u',str(S[1]/'src/refine_dynhamr.py'),'--input',out('01_track'),'--output',o,'--config',str(S[1]/'configs/dynhamr_camera_relative.json')]),
 ('02_targets',['retarget'],'02_targets',lambda o:[py(2),'-u',str(S[2]/'src/run_retarget.py'),'--input',out('01_dynhamr'),'--clip',str(C),'--output',o]),
 ('03_arm',['sam3'],'03_arm',lambda o:['bash',str(S[3]/'run_segment.sh'),'--clip',str(C),'--config',str(S[3]/'configs/arm.yaml'),'--output',o]),
 ('03_shadow',['sam3'],'03_shadow',lambda o:['bash',str(S[3]/'run_segment.sh'),'--clip',str(C),'--config',str(S[3]/'configs/shadow.yaml'),'--output',o]),
 ('03_prepared',['sam3'],'03_prepared',lambda o:[py(3),'-u',str(S[3]/'src/merge_shadow_masks.py'),'--arm-run',out('03_arm'),'--shadow-run',out('03_shadow'),'--output',o]),
 ('04_inpaint',['propainter'],'04_inpaint',lambda o:['bash',str(S[4]/'run_full.sh'),o,out('03_prepared')]),
 ('05_robot',['mujoco_mink'],'05_robot',lambda o:['bash',str(S[5]/'run.sh'),'--targets',out('02_targets'),'--background',out('04_inpaint'),'--config',str(S[5]/'configs/panda_production.yaml'),'--workers','8','--output',o]),
 ('05_validate',['mujoco_mink'],None,lambda o:[py(5),str(S[5]/'src/check_production.py'),'--output',out('05_robot')]),
 ('06_input_check',['da3'],None,lambda o:['bash',str(S[6]/'run.sh'),'--robot',out('05_robot'),'--clip',str(C),'--check-inputs']),
 ('06_composite',['da3'],'06_composite',lambda o:['bash',str(S[6]/'run.sh'),'--robot',out('05_robot'),'--clip',str(C),'--checkpoint',str(S[6]/'assets/DA3METRIC-LARGE'),'--output',o]),
 ('06_validate',['da3'],None,lambda o:[py(6),str(S[6]/'src/check_pipeline.py'),'--output',out('06_composite')]),
 ('06_depth_scale',['da3'],None,lambda o:[py(6),str(S[6]/'src/inspect_depth_scale.py'),'--output',out('06_composite')]),
]
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--only',nargs='+');args=a.parse_args()
 try:
  for label,envs,key,builder in JOBS:
   if not args.only or label in args.only:execute(label,envs,builder,key)
 except BaseException as exc:
  (T/'reports/last_pipeline_error.json').write_text(json.dumps({'time':now(),'error':str(exc)},indent=2));raise
 if all((T/'reports'/(j[0]+'.ok.json')).exists() for j in JOBS):
  status=read(T/'reports/status.json',{});status.update(status='all_six_stages_and_checks_completed',finished=now());(T/'reports/status.json').write_text(json.dumps(status,indent=2))
 print(now(),'requested stages completed',flush=True)
