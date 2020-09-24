'''
Test the slicing of data against extraction of whole data block
W Clarke, University of Oxford
2020
'''

import os.path as op
import numpy as np
from mapvbvd import mapVBVD

test_data_vb = op.join(op.dirname(__file__), 'test_data', 'meas_MID311_STEAM_wref1_FID115674.dat')
test_data_ve = op.join(op.dirname(__file__), 'test_data', 'meas_MID00305_FID74175_VOI_slaser_wref1.dat')
test_data_vb_broken = op.join(op.dirname(__file__), 'test_data', 'meas_MID111_sLaser_broken_FID4873.dat')


def test_vb():
    twixObj = mapVBVD(test_data_vb, quiet=False)
    twixObj.image.flagRemoveOS = False
    assert np.allclose(twixObj.image.fullSize, [4096, 32, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1])
    assert np.allclose(twixObj.image.sqzSize, [4096, 32, 2])

    twixObj.image.squeeze = True
    assert twixObj.image[0:2000,0:32:4,0:2].shape == (2000,8,2)
    assert twixObj.image[1000:2000,0:32:4,1].shape == (1000,8)
    assert twixObj.image[10,0:32:4,1].shape == (8,)

    fulldata = twixObj.image['']
    assert np.allclose(twixObj.image[1000:2000,0:32:4,1],fulldata[1000:2000,0:32:4,1])

    twixObj.image.squeeze = False
    assert twixObj.image[0:2000,0:32:4, :, :, :, :, :, :,0:2,:].shape == (2000,8, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1)
    assert np.allclose(np.squeeze(twixObj.image[1000:2000,0:32:4, :, :, :, :, :, :,1,:]),fulldata[1000:2000,0:32:4,1])