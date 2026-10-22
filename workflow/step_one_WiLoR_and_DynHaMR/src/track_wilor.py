"""Associate exported detections into two initial tracks without GPU inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from association import associate,jump_mask


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hand-masks',type=Path,help='Completed SAM3 A.1 hand mask run')
    a=p.parse_args(); source=a.input.resolve(); output=a.output.resolve()
    export=json.loads((source/'run.json').read_text())
    summary=json.loads((source/'summary.json').read_text())
    if export['schema']!='pkdata.wilor.detections.v1' or export['status']!='succeeded' or not summary['complete_clip']:
        p.error('input must be a completed full-clip WiLoR export')
    paths=sorted((source/'frames').glob('frame_*.npz'))
    assert len(paths)==summary['frames_processed']
    masks=a.hand_masks.resolve() if a.hand_masks else None
    if masks:
        mask_summary=json.loads((masks/'summary.json').read_text())
        if (mask_summary['schema']!='pkdata.sam3.hand_masks.v1' or
            mask_summary['status']!='succeeded' or not mask_summary['complete_clip'] or
            Path(mask_summary['source']).resolve()!=source):
            p.error('Hand masks must cover this exact completed WiLoR export')
    frames=[]
    for i,path in enumerate(paths):
        with np.load(path,allow_pickle=False) as archive:
            f={k:archive[k] for k in archive.files}
        assert int(f['frame_index'])==i
        assert np.array_equal(f['K'],np.asarray(export['K']))
        assert np.isin(f['is_right'],[0,1]).all()
        if masks:
            with np.load(masks/'frames'/path.name,allow_pickle=False) as gate:
                assert int(gate['frame_index'])==i
                assert np.array_equal(gate['detection_id'],f['detection_id'])
                f['sam3_mask_keep']=gate['mask_keep'].astype(bool)
                f['sam3_inside_count']=gate['inside_count'].astype(np.int32)
        frames.append(f)
    if output.exists() and any(output.iterdir()):
        p.error('output must be empty; existing results are preserved')
    selected,seed,seed_score=associate(frames)
    n=len(frames)
    output.mkdir(parents=True,exist_ok=True)
    for name in ['camera_parameters.json','frame_alignment.csv']:
        shutil.copy2(source/name,output/name)
    shutil.copy2(source/'run.json',output/'source_run.json')
    # Per-detection fields vs metadata are fixed by the exported schema.
    metadata_keys={'K','frame_index','rgb_source_frame_index','rgb_header_timestamp_ns','clip_video_time_s'}
    fields=[k for k in frames[0] if k not in metadata_keys]
    reject_rows=[]; selection_rows=[]; side_summaries={}; total_selected=0
    for side,name in enumerate(['left','right']):
        track={}
        for key in fields:
            example=frames[0][key]
            dtype=example.dtype
            fill=np.nan if dtype.kind=='f' else (False if dtype.kind=='b' else -1)
            array=np.full((n,*example.shape[1:]),fill,dtype=dtype)
            for t in range(n):
                if selected[t,side]>=0:
                    array[t]=frames[t][key][selected[t,side]]
            track[key]=array
        present=selected[:,side]>=0
        mismatch=present & (track['is_right']!=side)
        geometry_valid=present & ~mismatch & track['translation_fit_valid']
        jumps,threshold,velocity=jump_mask(track['keypoints_cam_pinhole_m'][:,0],geometry_valid)
        valid=geometry_valid & ~jumps
        track['keypoints_cam_filtered_m']=track['keypoints_cam_pinhole_m'].copy()
        track['keypoints_cam_filtered_m'][~valid]=np.nan
        track.update(handedness=np.array(name),source_row=selected[:,side],association_present=present,
                     handedness_mismatch=mismatch,jump_rejected=jumps,valid=valid,
                     wrist_step_m=velocity,jump_threshold_m=np.array(threshold),K=frames[0]['K'])
        for key in metadata_keys-{'K'}:
            track[key]=np.stack([f[key] for f in frames])
        np.savez_compressed(output/f'{name}_track.npz',**track)
        total_selected+=int(present.sum())
        z=track['keypoints_cam_pinhole_m'][:,0,2]
        side_summaries[name]=dict(associated_frames=int(present.sum()),valid_after_jump_filter=int(valid.sum()),
            missing_frames=np.flatnonzero(~present).tolist(),handedness_mismatch_frames=np.flatnonzero(mismatch).tolist(),
            invalid_translation_frames=np.flatnonzero(present & ~track['translation_fit_valid']).tolist(),
            jump_frames=np.flatnonzero(jumps).tolist(),jump_threshold_m_per_frame=threshold,
            wrist_depth_outside_paper_005_04_m=np.flatnonzero(geometry_valid & ((z<.05)|(z>.4))).tolist())
        for t in range(n):
            selection_rows.append(dict(frame_index=t,track=name,source_row=int(selected[t,side]),
                source_detection_id=int(track['detection_id'][t]),association_present=bool(present[t]),
                valid=bool(valid[t]),jump_rejected=bool(jumps[t]),handedness_mismatch=bool(mismatch[t])))
    for t,f in enumerate(frames):
        keep=set(selected[t][selected[t]>=0].tolist())
        for i in range(len(f['score'])):
            if i not in keep:
                reject_rows.append(dict(frame_index=t,source_row=i,detection_id=int(f['detection_id'][i]),
                    raw_side='right' if f['is_right'][i] else 'left',score=float(f['score'][i]),
                    reason=('sam3_hand_mask' if masks and not f['sam3_mask_keep'][i]
                            else 'not_selected_by_nearest_wrist_and_score')))
    for name,rows,columns in [
        ('selection.csv',selection_rows,['frame_index','track','source_row','source_detection_id','association_present','valid','jump_rejected','handedness_mismatch']),
        ('rejected_candidates.csv',reject_rows,['frame_index','source_row','detection_id','raw_side','score','reason'])]:
        with (output/name).open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    result=dict(schema='pkdata.wilor.association.v1',source=str(source),frames=n,
        seed_frame=seed,seed_combined_score=seed_score,total_raw_detections=summary['total_detections'],
        selected_detections=total_selected,rejected_candidates=len(reject_rows),tracks=side_summaries,
        tracking_applied=True,jump_filter_applied=True,sam3_mask_gate_applied=bool(masks),
        sam3_hand_masks=str(masks) if masks else None,
        sam3_mask_rejected_candidates=sum(int((~f['sam3_mask_keep']).sum()) for f in frames) if masks else 0,
        gap_filling_applied=False,dynhamr_applied=False,metric_depth_validated=False,
        note='Initial pinhole tracks. Jump flags require review; no MANO parameters or depth values are clamped.',
        code_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in
                     [Path(__file__),Path(__file__).with_name('association.py')]})
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));print('Output:',output)


if __name__=='__main__':
    main()
