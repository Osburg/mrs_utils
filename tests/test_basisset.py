import os

import numpy as np
import pytest

from mrs_utils.basisset import Basisset, io_readlcmraw_basis, io_readlcmraw

filepath = os.path.dirname(os.path.abspath(__file__)) + "/data/"


def test_io_readlcmraw_basis():
    out = io_readlcmraw_basis(filepath + "basisset.basis")

    assert len(out.keys()) == 26
    for k in out.keys():
        assert len(out[k]["fids"]) == 65458


def test_io_readlcmraw():
    out = io_readlcmraw(filepath + "Ala.RAW")

    assert list(out.keys()) == ["Ala_1.300000"]
    assert len(out["Ala_1.300000"]) == 65458 / 2


def test_basisset_basic():
    # correct usage
    fids = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
    names = ["m1", "m2", "m3", "m4"]
    basis = Basisset(fids, names, False, 1.0)

    assert basis.fids.shape == (3, 4)
    assert basis.spectra.shape == (3, 4)
    assert basis.metabolite_names == names
    assert np.allclose(basis.spectra[:, 0], [0, 0, 0])
    assert np.allclose(basis.spectra[:, 1], [0, 3, 0])
    assert np.allclose(basis.spectra[:, 2], [0, 6, 0])
    assert np.allclose(basis.spectra[:, 3], [0, 9, 0])

    # invalid metabolite name list
    with pytest.raises(ValueError):
        Basisset(fids, names + ["m5"], False, 1.0)

    # magic functions
    assert len(basis) == 4
    assert np.allclose(basis[1], [1, 1, 1])
    basis[2] *= 2
    for fid in basis:
        assert len(fid) == 3

    # conjugation
    basis[1] *= 1j
    basis.conjugate()
    assert basis.conjugate_basis
    assert np.allclose(basis[1], [-1j, -1j, -1j])
    assert np.allclose(basis[0], [0, 0, 0])
    assert np.allclose(basis[2], [4, 4, 4])
    assert np.allclose(basis[3], [3, 3, 3])

    assert basis.get_name_from_index(0) == "m1"


def test_Basisset_io():
    # from RAW
    basis = Basisset.from_RAW([filepath + "Ala.RAW"])
    assert not basis.conjugate_basis
    assert basis.metabolite_names == ["Ala_1.300000"]
    assert basis.fids.shape == (65458 / 2, 1)

    # from BASIS
    basis = Basisset.from_BASIS(filepath + "basisset.basis")
    assert basis.conjugate_basis
    assert basis.metabolite_names == [
        "Act",
        "Ala",
        "Glc",
        "Asp",
        "Glc_B",
        "Cho",
        "Cr",
        "GABA",
        "GPC",
        "GSH",
        "Gln",
        "Ins",
        "Lac",
        "MM_mea",
        "NAAG",
        "NAA",
        "PCh",
        "PCr",
        "Scyllo",
        "Tau",
        "TwoHG",
        "Glu",
        "Gly",
        "Lip_c",
        "mm3",
        "mm4",
    ]
    assert basis.fids.shape == (65458, 26)
    assert np.allclose(basis.dwelltime, 3.3333e-5)

    # from npz
    metabolite_names = [f"m{i}" for i in range(19)]
    basis = Basisset.from_npz(filepath + "basisset.npz", metabolite_names)
    assert basis.fids.shape == (2048, 19)
    assert basis.metabolite_names == metabolite_names

    # from matlab
    basis = Basisset.from_matlab(filepath + "basisset.mat", metabolite_names)
    assert basis.fids.shape == (2048, 19)
    assert basis.metabolite_names == metabolite_names
