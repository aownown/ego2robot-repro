"""Behavioral checks for masking, BMC units/chirality, and rotation gap filling."""
import argparse
import contextlib
import io
import json
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from dynhamr_adapter import STEP, build_input, make_model, tensor_dict, interpolate_initialization
from dynhamr_a1_loss import A1Loss, canonical_bmc_joints
from optim.base_scene import BaseSceneModel
from util.logger import Logger


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('tracks')
    a=p.parse_args()
    cfg=json.loads((STEP/'configs/dynhamr_camera_relative.json').read_text())
    data,cam,meta,_=build_input(a.tracks,0,90)
    assert not data['observed_valid'][1,58:60].any()
    assert np.isfinite(data['init_trans']).all()
    assert (data['vis_mask'][~data['observed_valid']]<0).all()
    assert (data['joints2d'][~data['observed_valid'],:,2]==0).all()
    Logger.init('/tmp/pkdata-dynhamr-adapter-check.log')
    with contextlib.redirect_stdout(io.StringIO()):
        model=make_model(meta['frames'])
        obs,cam_t=tensor_dict(data),tensor_dict(cam)
        scene=BaseSceneModel(2,meta['frames'],model,None,use_init=True,opt_cams=False,opt_scale=False).cuda()
        scene.initialize(obs,cam_t)
        scene.params.set_require_grads(['trans','root_orient','betas','latent_pose'])
        pred=scene.pred_params_mano(obs['is_right'])
        pred['cameras']=scene.params.get_cameras()
        pred.update(scene.params.get_vars())
        loss_fn=A1Loss(cfg,True)
        loss,stats=loss_fn(obs,pred)
        altered={k:v.clone() for k,v in obs.items()}
        for key in ('joints2d','joints3d','raw_mano_rotations'):
            altered[key][~obs['observed_valid']]=10000
        other,_=loss_fn(altered,pred)
        torch.testing.assert_close(loss,other,rtol=0,atol=0)
        loss.backward()
        assert all(v.grad is None or torch.isfinite(v.grad).all() for v in scene.parameters())
        assert scene.params.trans.grad.abs().sum()>0
        # BMC input stays invariant under metric scaling and translation.
        joints=pred['joints3d'].detach()
        normalized=canonical_bmc_joints(joints,obs['is_right'])
        scaled=canonical_bmc_joints(joints*3.7+.2,obs['is_right'])
        torch.testing.assert_close(normalized,scaled,atol=3e-6,rtol=3e-6)
        right=joints[1:2]
        left=right*torch.tensor([-1.,1.,1.],device='cuda')
        canonical_left=canonical_bmc_joints(left,torch.zeros_like(obs['is_right'][1:2]))
        canonical_right=canonical_bmc_joints(right,torch.ones_like(obs['is_right'][1:2]))
        torch.testing.assert_close(canonical_left,canonical_right)
    # 179 -> -179 degrees must interpolate through 180, not through zero.
    rot=Rotation.from_euler('z',[179,0,-179],degrees=True).as_matrix()[:,None]
    pos=np.array([[0.,0,0],[99,99,99],[2,0,0]])
    trans,filled,_=interpolate_initialization(pos,rot,np.array([True,False,True]),10)
    np.testing.assert_allclose(trans[1],[1,0,0])
    assert abs(abs(Rotation.from_matrix(filled[1,0]).as_rotvec()[2])-np.pi)<1e-6
    try:
        interpolate_initialization(pos,rot,np.array([False,True,True]),10)
    except ValueError:
        pass
    else:
        raise AssertionError('Boundary gaps must not be silently extrapolated')
    print('Passed: invalid-observation exclusion, finite nonzero gradients, BMC scale/chirality, SLERP across pi, boundary-gap rejection')


if __name__=='__main__':
    main()
