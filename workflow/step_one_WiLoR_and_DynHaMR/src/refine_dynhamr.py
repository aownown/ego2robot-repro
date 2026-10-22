"""Run a documented camera-relative Ego2Robot A.1 adaptation of Dyn-HaMR."""
import argparse
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
import torch
import yaml

from dynhamr_adapter import STEP, REPO, build_input, make_model, tensor_dict, validate_mano_mapping
from dynhamr_a1_loss import A1Loss, canonical_bmc_joints
from optim.base_scene import BaseSceneModel
from optim.optimizers import RootOptimizer, SmoothOptimizer
from util.logger import Logger


def snapshot(scene):
    return {k:v.detach().clone() for k,v in scene.params.get_vars().items()}


def restore(scene, saved):
    with torch.no_grad():
        for k,v in saved.items():
            scene.params.get_param(k).copy_(v)


def metrics(points, uv_target, K, raw_points, valid, depth_target='wrist'):
    projected=points@K.T
    projected=projected[...,:2]/projected[...,2:]
    err=np.linalg.norm(projected-uv_target,axis=-1)
    wrist=points[:,0]
    acc=np.diff(wrist,n=2,axis=0)
    depth=points[:,0,2] if depth_target=='wrist' else points[...,2]
    outside=(depth<.05)|(depth>.4)
    if outside.ndim>1:
        outside=outside.any(-1)
    return dict(reprojection_rmse_px=float(np.sqrt(np.mean(err[valid]**2))),
                wrist_acceleration_rms_m_per_frame2=float(np.sqrt(np.mean(np.sum(acc**2,axis=-1)))),
                wrist_step_max_m=float(np.linalg.norm(np.diff(wrist,axis=0),axis=-1).max()),
                wrist_change_from_wilor_rms_m=float(np.sqrt(np.mean(np.sum((wrist[valid]-raw_points[valid,0])**2,axis=-1)))),
                joint_depth_min_m=float(points[...,2].min()),joint_depth_max_m=float(points[...,2].max()),
                wrist_depth_min_m=float(wrist[:,2].min()),wrist_depth_max_m=float(wrist[:,2].max()),
                depth_target=depth_target,frames_outside_depth_range=np.flatnonzero(outside).tolist(),
                depth_violation_max_m=float(np.maximum(.05-depth,np.maximum(depth-.4,0)).max()))


def draw_skeleton(image, points, color):
    if not np.isfinite(points).all():
        return
    xy=np.rint(points).astype(int)
    for finger in range(5):
        ids=[0]+list(range(1+4*finger,5+4*finger))
        for a,b in zip(ids[:-1],ids[1:]):
            cv2.line(image,tuple(xy[a]),tuple(xy[b]),color,2,cv2.LINE_AA)
    for point in xy:
        cv2.circle(image,tuple(point),3,color,-1,cv2.LINE_AA)


def render_comparison(output, raw, optimized, meta, export):
    clip=Path(export['input'])
    with (clip/'frame_alignment.csv').open() as f:
        rows=list(csv.DictReader(f))
    first=cv2.imread(str(clip/rows[meta['start_frame']]['rgb_file']))
    h,w=first.shape[:2]
    preview=output/'comparison.mp4'
    proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','bgr24',
                           '-video_size',f'{w*2}x{h}','-framerate',str(export['fps']),'-i','-',
                           '-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',str(preview)],stdin=subprocess.PIPE)
    K=np.asarray(meta['K'])
    (output/'preview_samples').mkdir()
    try:
        for t in range(meta['frames']):
            i=t+meta['start_frame']
            image=cv2.imread(str(clip/rows[i]['rgb_file']))
            before,after=image.copy(),image.copy()
            for side in range(2):
                color=(255,255,0) if side==0 else (0,220,255)
                p=raw[side]['keypoints_cam_pinhole_m'][i]@K.T
                draw_skeleton(before,p[:,:2]/p[:,2:],color)
                p=optimized[side,t]@K.T
                draw_skeleton(after,p[:,:2]/p[:,2:],color)
                if not raw[side]['valid'][i]:
                    cv2.putText(after,f'{["L","R"][side]}: inferred (observation excluded)',(15,70+side*30),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
            cv2.putText(before,f'WiLoR pinhole init | frame {i}',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
            cv2.putText(after,'DynHaMR A.1 adaptation | camera-relative',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
            canvas=np.concatenate([before,after],axis=1)
            proc.stdin.write(canvas.tobytes())
            if t%30==0 or i in (14,45,58,59,100,144,155,221,232,325) or t==meta['frames']-1:
                cv2.imwrite(str(output/'preview_samples'/f'frame_{i:06d}.jpg'),canvas)
    finally:
        proc.stdin.close()
        code=proc.wait()
    if code:
        raise RuntimeError(f'ffmpeg exited {code}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=STEP/'configs/dynhamr_camera_relative.json')
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--root-iters',type=int)
    p.add_argument('--smooth-iters',type=int)
    a=p.parse_args()
    cfg=json.loads(a.config.read_text())
    for attr,key in [('root_iters','root_iterations'),('smooth_iters','smooth_iterations')]:
        if getattr(a,attr) is not None:
            cfg[key]=getattr(a,attr)
    if min(cfg['root_iterations'],cfg['smooth_iterations'])<1:
        p.error('Both optimization stages require >=1 iteration')
    output=a.output.resolve()
    if output.exists() and any(output.iterdir()):
        p.error('Output must be empty')
    output.mkdir(parents=True,exist_ok=True)
    data,cam,meta,raw=build_input(a.input,a.start,a.limit,cfg['max_initial_gap_frames'])
    n=meta['frames'];start=meta['start_frame'];end=meta['end_frame_exclusive']
    source=Path(meta['source'])
    export=json.loads((source/'source_run.json').read_text())
    torch.manual_seed(42)
    torch.set_num_threads(4)
    np.savez_compressed(output/'optimizer_input.npz',**data,**{'camera_'+k:v for k,v in cam.items()})
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    meta.update(schema='pkdata.dynhamr.camera_relative.v1',status='running',profile=cfg['profile'],
                upstream_revision=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
                exact_paper_reproduction=False,world_trajectory_recovered=False,metric_depth_validated=False,
                hmp_applied=False,depth_constraint_type='soft penalty',depth_target=cfg['depth_target'],
                initial_gap_interpolation=True,final_gap_interpolation=False,
                loss_implementation='Local A.1 adapter with recorded weights; upstream MANO, scene, stage parameter sets, LBFGS and BMC',
                code_sha256={name:hashlib.sha256((STEP/'src'/name).read_bytes()).hexdigest() for name in
                             ['refine_dynhamr.py','dynhamr_adapter.py','dynhamr_a1_loss.py']})
    (output/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    Logger.init(str(output/'upstream.log'))
    with contextlib.redirect_stdout(io.StringIO()):
        model=make_model(n)
        mapping=validate_mano_mapping(model,raw,start,end)
        obs,cam_tensors=tensor_dict(data),tensor_dict(cam)
        scene=BaseSceneModel(2,n,model,None,use_init=True,opt_cams=False,opt_scale=False).cuda()
        scene.initialize(obs,cam_tensors)
    print('MANO mapping passed:',mapping,flush=True)
    with torch.no_grad(),contextlib.redirect_stdout(io.StringIO()):
        initial=scene.pred_params_mano(obs['is_right'])['joints3d'].cpu().numpy()
    # Import the native optimizer parameter grouping/LBFGS; replace its loss
    # explicitly instead of silently modifying the checked-out source.
    upstream_cfg=yaml.safe_load((REPO/'dyn-hamr/confs/optim.yaml').read_text())['optim']
    weights=[{k:v[i] for k,v in upstream_cfg['loss_weights'].items()} for i in range(3)]
    history=[]; phases={}; begin=time.monotonic()
    for cls,count,smooth in [(RootOptimizer,cfg['root_iterations'],False),(SmoothOptimizer,cfg['smooth_iterations'],True)]:
        with contextlib.redirect_stdout(io.StringIO()):
            optimizer=cls(scene,weights,lr=cfg['lr'],lbfgs_max_iter=cfg['lbfgs_max_iter'],
                          save_meshes=False,vis_every=-1)
            bmc=optimizer.loss.bio_loss
            optimizer.loss=A1Loss(cfg,smooth,bmc)
        phase=optimizer.name
        best=snapshot(scene)
        with torch.no_grad(),contextlib.redirect_stdout(io.StringIO()):
            initial_loss,_,_=optimizer.forward_pass(obs)
        best_loss=float(initial_loss)
        if not np.isfinite(best_loss):
            raise FloatingPointError(f'{phase}: nonfinite initial objective')
        phases[phase]=dict(initial_loss=best_loss,requested_iterations=count)
        stable=0
        for iteration in range(count):
            before=snapshot(scene)
            calls=[0]
            def closure():
                calls[0]+=1
                optimizer.optim.zero_grad()
                with contextlib.redirect_stdout(io.StringIO()):
                    loss,stats,_=optimizer.forward_pass(obs)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite objective')
                loss.backward()
                if any(param.grad is not None and not torch.isfinite(param.grad).all() for param in optimizer.opt_params):
                    raise FloatingPointError('Nonfinite gradient')
                return loss
            try:
                optimizer.optim.step(closure)
            except FloatingPointError:
                restore(scene,before)
                raise
            with torch.no_grad(),contextlib.redirect_stdout(io.StringIO()):
                loss,stats,_=optimizer.forward_pass(obs)
            current=float(loss)
            if not np.isfinite(current):
                restore(scene,best)
                raise FloatingPointError('Nonfinite post-step objective')
            previous=best_loss
            if current<=best_loss:
                best=snapshot(scene);best_loss=current
            if current>previous+max(1e-6,abs(previous)*1e-5):
                restore(scene,best)
                phases[phase]['stop_reason']='objective_increase_restored_best'
                break
            history.append(dict(phase=phase,iteration=iteration+1,loss=current,closure_calls=calls[0],
                                **{k:float(v) for k,v in stats.items()}))
            if iteration%5==0 or iteration==count-1:
                print(f'{phase} {iteration+1}/{count}: loss={current:.6f} | {time.monotonic()-begin:.1f}s',flush=True)
            stable=stable+1 if previous-current<max(1e-8,abs(previous)*1e-7) else 0
            if stable>=3:
                phases[phase]['stop_reason']='objective_converged'
                break
        restore(scene,best)
        phases[phase].update(final_loss=best_loss,completed_iterations=iteration+1)
        torch.save({k:v.cpu() for k,v in best.items()},output/f'{phase}_params.pt')
        (output/'loss_history.json').write_text(json.dumps(history,indent=2)+'\n')
    with torch.no_grad(),contextlib.redirect_stdout(io.StringIO()):
        pred=scene.pred_params_mano(obs['is_right'])
        optimized=pred['joints3d'].cpu().numpy()
        vertices=pred['verts3d'].cpu().numpy()
        params={k:v.detach().cpu().numpy() for k,v in scene.params.get_vars().items()}
    if not np.isfinite(optimized).all() or not np.isfinite(vertices).all():
        raise FloatingPointError('Final geometry is not finite')
    reports={}
    K=np.asarray(meta['K'])
    for side,name in enumerate(('left','right')):
        f=raw[side];valid=data['observed_valid'][side]
        raw_points=f['keypoints_cam_pinhole_m'][start:end]
        target=f['keypoints_2d_model_px'][start:end]
        reports[name]={label:metrics(points,target,K,raw_points,valid,cfg['depth_target']) for label,points in
                       [('wilor',raw_points),('shared_shape_initialization',initial[side]),('optimized',optimized[side])]}
        trans=params['trans'][side].copy()
        trans[:,0]*=2*side-1
        fields=dict(keypoints_cam_m=optimized[side],vertices_cam_m=vertices[side],
                    mano_global_orient_axis_angle=params['root_orient'][side],
                    mano_hand_pose_axis_angle=params['latent_pose'][side].reshape(n,15,3),
                    mano_betas=params['betas'][side],camera_translation_m=trans,
                    observed_valid=valid,optimization_inferred=~valid,
                    finite=np.isfinite(optimized[side]).all(axis=(1,2)),
                    handedness=name,K=K,source_row=f['source_row'][start:end],
                    keypoints_cam_wilor_init_m=raw_points)
        for key in ['frame_index','clip_video_time_s','rgb_header_timestamp_ns','rgb_source_frame_index']:
            fields[key]=f[key][start:end]
        np.savez_compressed(output/f'{name}_track.npz',**fields)
    for name in ('camera_parameters.json','frame_alignment.csv'):
        shutil.copy2(source/name,output/name)
    render_comparison(output,raw,optimized,meta,export)
    meta.update(status='succeeded',dynhamr_applied=True,frames_optimized=n,all_outputs_finite=True,
                mano_mapping=mapping,phases=phases,metrics=reports,elapsed_seconds=time.monotonic()-begin,
                observed_frames={name:int(data['observed_valid'][s].sum()) for s,name in enumerate(('left','right'))},
                note='Camera-relative optimization prototype. Data terms exclude flagged observations; smoothness/BMC/depth also constrain inferred frames. Review before retargeting.')
    for name in ('run.json','summary.json'):
        (output/name).write_text(json.dumps(meta,indent=2)+'\n')
    print('Output:',output,flush=True)


if __name__=='__main__':
    main()
