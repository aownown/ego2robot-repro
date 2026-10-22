import numpy as np
from camera_init import crop_to_pixels, fit_translation, project

rng=np.random.default_rng(24)
K=np.array([[692.8,0,644.6],[0,692.4,365.8],[0,0,1.]])
xyz=rng.uniform(-.05,.05,(21,3))
truth=np.array([.07,-.03,.45])
for side in (0,1):
    points=xyz.copy(); points[:,0]*=2*side-1
    pixels=project(points+truth,K)
    translation,rmse,ok=fit_translation(points,pixels,K)
    assert ok and rmse<1e-6
    np.testing.assert_allclose(translation,truth,atol=1e-8)
    center=np.array([400.,250.]); size=180.
    crop=(pixels-center)/size; crop[:,0]*=2*side-1
    np.testing.assert_allclose(crop_to_pixels(crop,center,size,side),pixels)
bad=xyz.copy(); bad[0]=np.nan
assert not fit_translation(bad,project(xyz+truth,K),K)[2]
print('Known-camera translation recovery, left/right crop inversion and invalid input checks passed')
