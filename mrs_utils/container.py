import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, TensorDataset
from typing_extensions import Self
from torchvision.transforms import ToTensor
from torch import from_numpy, Tensor
from typing import List
from overrides import override

class MRSContainer(TensorDataset):
    """Class to store mrs(i) data"""

    def __init__(self, data: np.ndarray, **kwargs) -> None:
        """Initializes the Container object which inherits from the Dataset class
        
        Args:
            data (np.ndarray): The data to be stored in the container. 
                The samples should be stored along the first dimension.
            **kwargs: Additional keyword arguments to be stored in the container
                device (str): The device to store the data on. Default is "cpu"
                dwelltime (float): The dwell time of the data. Default is None
                reference_frequency (float): The reference frequency of the data. Default is None
                transform (callable): The transform to be applied to the data. Default is lambda x: x.
                This can for example be used to include an augmentation step.
        
        """
        
        if "device" in kwargs:
            self.device = kwargs.get("device")
        else:
            self.device = "cpu"

        if "dwelltime" in kwargs:
            self.dwelltime = kwargs.get("dwelltime")
        else:
            self.dwelltime = None
        
        if "reference_frequency" in kwargs:
            self.reference_frequency = kwargs.get("reference_frequency")
        else:
            self.reference_frequency = None

        if "transform" in kwargs:
            self.transform = kwargs.get("transform")
        else:
            self.transform = lambda x: x

        super().__init__(from_numpy(data.astype("complex64")).to(self.device))
        self.n_signals = data.shape[0]
        self.signal_length = data.shape[1]

    @override
    def __getitem__(self, idx: int) -> Tensor:
        return tuple(self.transform(tensor[idx]) for tensor in self.tensors)

    @classmethod
    def from_matlab(cls, path: str, **kwargs) -> Self:
        data = sio.loadmat(path).get('data').T
        return cls(data, **kwargs)
    
    @classmethod
    def from_npz(cls, path: str, **kwargs) -> Self:
        data = np.load(path)
        data = data["data"]
        return cls(data, **kwargs)

    # TODO: test
    @classmethod
    def from_npy(cls, path: str, **kwargs) -> Self:
        data = np.load(path)
        return cls(data, **kwargs)


    # TODO: test
    def to_npy(self, path: str) -> None:
        np.save(path, self.tensors[0].detach().cpu().numpy())

    def to_npz(self, path: str) -> None:
        np.savez(path, data=self.tensors[0].detach().cpu().numpy())

    def to_matlab(self, path: str) -> None:
        dict = {
            "__header__": b"MAT-file",
            "__version__": "1.0",
            "__globals__": [],
            "data": self.tensors[0].detach().cpu().numpy().T,
            }
        sio.savemat(path, dict)

    # TODO: implement, test
    def remove_water_signal(self):
        """Apply HLSVD to remove the water signal from the data"""
        NotImplementedError
