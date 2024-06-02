from utils.container import MRSContainer
import numpy as np
import os


def test_container():
    # test constructor
    container = MRSContainer(
        data=np.array([[1,2,3,4]]),
        device="cpu",
        dwell_time=1,
        reference_frequency=1,
        transform=lambda x: x
    )
    assert len(container) == 1
    assert np.allclose(container[0].detach().cpu().numpy(), np.array([1,2,3,4]))
    assert container.device == "cpu"
    assert container.dwell_time == 1
    assert container.reference_frequency == 1
    assert np.allclose(container.transform(container[0]), container[0])

    # test reading and writing containers from/to matlab
    container = MRSContainer.from_matlab(
        os.path.dirname(os.path.realpath(__file__)) +  "/data/basisset.mat")
    assert len(container) == 19
    assert container.device == "cpu"
    assert container.signal_length == 2048
    container.to_matlab(
        os.path.dirname(os.path.realpath(__file__)) +  "/data/basisset_out.mat")
    container_reload = MRSContainer.from_matlab(
        os.path.dirname(os.path.realpath(__file__)) +  "/data/basisset_out.mat")
    assert len(container_reload) == 19
    assert container_reload.device == "cpu"
    assert container_reload.signal_length == 2048

    # test reading and writing containers from/to npz
    container = MRSContainer.from_npz(
        os.path.dirname(os.path.realpath(__file__)) +  "/data/basisset.npz")
    assert len(container) == 19
    assert container.device == "cpu"
    assert container.signal_length == 2048
    container.to_npz(
        os.path.dirname(os.path.realpath(__file__)) +  "/data/basisset_out.npz")
    container_reload = MRSContainer.from_npz(
        os.path.dirname(os.path.realpath(__file__)) +  "/data/basisset_out.npz")
    assert len(container_reload) == 19
    assert container_reload.device == "cpu"
    assert container_reload.signal_length == 2048