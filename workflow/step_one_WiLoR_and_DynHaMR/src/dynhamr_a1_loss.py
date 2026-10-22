"""Explicit local A.1 loss adapter around Dyn-HaMR's scene and BMC.

The paper does not publish all weights/normalizations. This profile records
local choices; it is not the unmodified upstream objective or an exact rerun.
"""
import contextlib
import io

import torch
from torch import nn
from torch.nn import functional as F
from smplx.lbs import batch_rodrigues
from optim.bio_loss import BMCLoss


def canonical_bmc_joints(joints, is_right):
    sign=torch.ones_like(joints[...,:1,:])
    sign[...,0]=(2*is_right-1)[...,None]
    canonical=joints*sign
    canonical=canonical-canonical[:,:,9:10]
    scale=torch.linalg.vector_norm(canonical[:,:,0]-canonical[:,:,9],dim=-1).clamp_min(1e-6)
    return canonical/scale[...,None,None]


def rotmats(aa):
    return batch_rodrigues(aa.reshape(-1,3)).reshape(*aa.shape[:-1],3,3)


class A1Loss(nn.Module):
    def __init__(self, cfg, smooth, bmc=None):
        super().__init__()
        self.cfg=cfg
        self.smooth=smooth
        self.bmc=bmc if bmc is not None else BMCLoss(lambda_bl=1.,lambda_rb=1.,lambda_a=1.)

    def forward(self, observed, pred, *args):
        # Native RootLoss receives mask; native SMPLLoss receives nsteps, mask.
        valid=observed['observed_valid'].bool()
        joints=pred['joints3d']
        cam_R,cam_t,cam_f,center=pred['cameras']
        camera=torch.einsum('btij,btkj->btki',cam_R,joints)+cam_t[:,:,None]
        uv=camera[...,:2]/camera[...,2:].clamp_min(1e-5)*cam_f[None,:,None]+center[None,:,None]
        scales=self.cfg['scales'];weights=self.cfg['weights']
        confidence=observed['joints2d'][...,2:3][valid]
        stats={}
        stats['data_2d']=(F.smooth_l1_loss(uv[valid]/scales['data_2d_px'],observed['joints2d'][...,:2][valid]/scales['data_2d_px'],reduction='none')*confidence.square()).mean()
        stats['data_3d']=((joints[valid]-observed['joints3d'][valid])/scales['data_3d_m']).square().mean()
        wrist=joints[:,:,0]
        accel=wrist[:,2:]-2*wrist[:,1:-1]+wrist[:,:-2]
        stats['wrist_acceleration']=(accel/scales['wrist_acceleration_m_per_frame2']).square().mean()
        # A.1 does not identify which point defines "hand depth". This profile
        # explicitly uses wrist depth, consistently with the trajectory audit.
        # Fingertips may extend beyond the wrist's allowed interval.
        depth=camera[:,:,0,2] if self.cfg['depth_target']=='wrist' else camera[...,2]
        stats['depth']=((F.relu(self.cfg['depth_min_m']-depth)+F.relu(depth-self.cfg['depth_max_m']))/scales['depth_m']).square().mean()
        if self.smooth:
            hand_aa=pred['body_pose'].reshape(*joints.shape[:2],15,3)
            hand_R=rotmats(hand_aa)
            root_R=rotmats(pred['root_orient'])[:,:,None]
            full_R=torch.cat([root_R,hand_R],dim=2)
            stats['pose_data']=(full_R[valid]-observed['raw_mano_rotations'][valid]).square().mean()
            stats['shape_data']=(pred['betas']-observed['shared_betas_target']).square().mean()
            # Change in relative joint rotation between successive frame pairs;
            # continuous through axis-angle +/-pi and vanishes at rest.
            delta=hand_R[:,1:]@hand_R[:,:-1].transpose(-1,-2)
            stats['finger_acceleration']=((delta[:,1:]-delta[:,:-1])/scales['finger_rotation_acceleration']).square().mean()
            # BMC statistics were generated after normalizing wrist->middle MCP
            # length to one; applying them directly to meters is incorrect.
            normalized=canonical_bmc_joints(joints,pred['is_right'])
            with contextlib.redirect_stdout(io.StringIO()):
                bio,_=self.bmc.compute_loss(normalized,torch.ones_like(valid))
            stats['bio']=bio.squeeze()
        loss=sum(weights[name]*value for name,value in stats.items())
        return loss,stats
