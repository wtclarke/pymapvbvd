"""
Created by shahdloo
22/09/2020
"""

import os.path as op
import numpy as np
from core import mapVBVD

test_data_vb_broken = op.join(op.dirname(__file__), 'test_data', 'meas_MID111_sLaser_broken_FID4873.dat')
test_data_gre = op.join(op.dirname(__file__), 'test_data', 'meas_MID00058_FID12358_gre_3D.dat')

def test_flagRemoveOS():
    twixObj = mapVBVD(test_data_gre, quiet=False)
    twixObj[1].image.flagRemoveOS = False

    assert np.allclose(twixObj.image.fullSize, [4096, 32, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1])
