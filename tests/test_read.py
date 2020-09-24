"""
Created by shahdloo
23/09/2020
"""

import os.path as op
import numpy as np
from mapvbvd import mapVBVD
import h5py

test_data_gre = op.join(op.dirname(__file__), 'test_data', 'meas_MID00255_FID12798_GRE_surf.dat')
test_data_gre_mat = op.join(op.dirname(__file__), 'test_data', 'meas_MID00255_FID12798_GRE_surf.mat')
test_data_epi = op.join(op.dirname(__file__), 'test_data', 'meas_MID00265_FID12808_FMRI.dat')
test_data_epi_mat = op.join(op.dirname(__file__), 'test_data', 'meas_MID00265_FID12808_FMRI.mat')


def test_gre():
    twixObj = mapVBVD(test_data_gre, quiet=False)
    twixObj[1].image.squeeze = True
    twixObj[1].image.flagRemoveOS = False
    img_py = twixObj[1].image[:, :, :, 0]
    twixObj[1].image.flagRemoveOS = True
    img_py_os = twixObj[1].image[:, :, :, 0]

    with h5py.File(test_data_gre_mat, 'r') as f:
        base = f['img'][0, 0, :, :, :]
        img_mat = (base['real'] + 1j * base['imag']).transpose()
        base = f['img_os'][0, 0, :, :, :]
        img_mat_os = (base['real'] + 1j * base['imag']).transpose()

    assert np.allclose(img_py, img_mat)
    assert np.allclose(img_py_os, img_mat_os)


def test_epi():
    twixObj = mapVBVD(test_data_epi, quiet=False)
    twixObj[1].image.squeeze = True
    twixObj[1].image.flagRampSampRegrid = False
    twixObj[1].image.flagRemoveOS = False
    img_py = twixObj[1].image[:, :, :, 0, 0, 0]
    twixObj[1].image.flagRemoveOS = True
    img_py_os = twixObj[1].image[:, :, :, 0, 0, 0]
    twixObj[1].image.flagRampSampRegrid = True
    img_py_os_rg = twixObj[1].image[:, :, :, 0, 0, 0]

    with h5py.File(test_data_epi_mat, 'r') as f:
        base = f['img'][0, 0, 0, 0, 0, 0, 0, 0, :, :, :]
        img_mat = (base['real'] + 1j * base['imag']).transpose()
        base = f['img_os'][0, 0, 0, 0, 0, 0, 0, 0, :, :, :]
        img_mat_os = (base['real'] + 1j * base['imag']).transpose()
        base = f['img_os_rg'][0, 0, 0, 0, 0, 0, 0, 0, :, :, :]
        img_mat_os_rg = (base['real'] + 1j * base['imag']).transpose()

    assert np.allclose(img_py, img_mat)
    assert np.allclose(img_py_os, img_mat_os)
    assert np.allclose(img_py_os_rg, img_mat_os_rg, atol=1e-6, rtol=1e-6)
