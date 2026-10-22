import numpy as np
from association import associate,jump_mask


def frame(xs,sides,scores):
    points=np.zeros((len(xs),21,2));points[:,:,0]=np.array(xs)[:,None]
    return dict(keypoints_2d_model_px=points,is_right=np.array(sides),score=np.array(scores))

# Seed at middle; distractor claims R but is spatially close to L. Rows reorder.
frames=[frame([10,90,13],[0,1,1],[.8,.8,.3]),frame([11,89],[0,1],[.99,.99]),
        frame([88,12,14],[1,0,1],[.8,.8,.3])]
selected,seed,_=associate(frames)
assert seed==1
np.testing.assert_array_equal(selected,[[0,1],[0,1],[1,0]])
# Missing observations remain missing and do not shift frame indices.
missing=frame([],[],[])
selected,_,_=associate([frames[0],frames[1],missing,frames[2]])
np.testing.assert_array_equal(selected[2],[-1,-1])
np.testing.assert_array_equal(selected[3],[1,0])
# Mask gating happens before seed choice and association, even for a high score.
gated=frame([10,90,11],[0,1,0],[.7,.8,.99])
gated['sam3_mask_keep']=np.array([True,True,False])
selected,_,_=associate([gated,frames[1],frames[2]])
assert selected[0,0]==0
# A one-frame spike flags both high-displacement later endpoints, without fill.
w=np.zeros((9,3));w[:,0]=np.arange(9)*.001;w[4,2]=.05
bad,threshold,_=jump_mask(w,np.ones(9,bool))
np.testing.assert_array_equal(np.flatnonzero(bad),[4,5])
assert abs(threshold-.004)<1e-10
print('Bidirectional association, row reordering, duplicate labels, missing frames and jump checks passed')
