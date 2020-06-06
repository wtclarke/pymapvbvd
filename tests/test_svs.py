import os.path as op
import numpy as np
from mapVBVD import mapVBVD

test_data_vb = op.join(op.dirname(__file__),'test_data','meas_MID311_STEAM_wref1_FID115674.dat')
test_data_ve = op.join(op.dirname(__file__),'test_data','meas_MID00305_FID74175_VOI_slaser_wref1.dat')


def test_vb():
    twixObj = mapVBVD(test_data_vb,quiet=True)
    assert np.allclose(twixObj.image.fullSize,[4096,32,1,1,1,1,1,1,2,1,1,1,1,1,1,1])
    assert np.allclose(twixObj.image.sqzSize(),[4096,32,2])


    keys = twixObj.search_header_for_keys(('sTXSPEC', 'asNucleusInfo'),top_lvl='MeasYaps')
    assert ('sTXSPEC', 'asNucleusInfo', '0', 'tNucleus') in keys['MeasYaps']
    key_value = twixObj.search_header_for_val('MeasYaps',('sTXSPEC', 'asNucleusInfo', '0', 'tNucleus'))
    assert key_value[0] == '"1H"'

def test_ve():
    twixObj = mapVBVD(test_data_ve,quiet=True)
    assert len(twixObj)==2
    twixObj[1].image

    assert np.allclose(twixObj[1].image.fullSize,[4096,32,1,1,1,1,1,1,2,1,1,1,1,1,1,1])
    assert np.allclose(twixObj[1].image.sqzSize(),[4096,32,2])

    keys = twixObj[1].search_header_for_keys(('sTXSPEC', 'asNucleusInfo'),top_lvl='MeasYaps')
    assert ('sTXSPEC', 'asNucleusInfo', '0', 'tNucleus') in keys['MeasYaps']
    key_value = twixObj[1].search_header_for_val('MeasYaps',('sTXSPEC', 'asNucleusInfo', '0', 'tNucleus'))
    assert key_value[0] == '"1H"'