import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from typing_extensions import Self
from torchvision.transforms import ToTensor
from torch import from_numpy, Tensor
from typing import List

class MRSContainer(Dataset):
    """Class to store mrs(i) data"""

    def __init__(self, data: np.ndarray, **kwargs) -> None:
        """Initializes the Container object which inherits from the Dataset class
        
        Args:
            data (np.ndarray): The data to be stored in the container. 
                The samples should be stored along the first dimension.
            **kwargs: Additional keyword arguments to be stored in the container
        
        """
        
        if "device" in kwargs:
            self.device = kwargs.get("device")
        else:
            self.device = "cpu"

        if "dwell_time" in kwargs:
            self.dwell_time = kwargs.get("dwell_time")
        else:
            self.dwell_time = None
        
        if "reference_frequency" in kwargs:
            self.reference_frequency = kwargs.get("reference_frequency")
        else:
            self.reference_frequency = None

        if "transform" in kwargs:
            self.transform = kwargs.get("transform")
        else:
            self.transform = lambda x: x


        self.data = from_numpy(data.astype("complex64")).to(self.device)
        self.n_signals = data.shape[0]
        self.signal_length = data.shape[1]

    def __len__(self) -> int:
        return self.n_signals
    
    def __getitem__(self, idx: int) -> Tensor:
        return self.transform(self.data[idx,:])
    
    def __iter__(self):
        return self.data.__iter__()
    
    def __next__(self):
        return self.data.__next__()

    @classmethod
    def from_matlab(cls, path: str) -> Self:
        data = sio.loadmat(path).get('data').T
        return cls(data)
    
    @classmethod
    def from_npz(cls, path: str) -> Self:
        data = np.load(path)
        data = data["data"]
        return cls(data)
    
    def to_npz(self, path: str) -> None:
        np.savez(path, data=self.data)

    def to_matlab(self, path: str) -> None:
        dict = {
            "__header__": b"MAT-file",
            "__version__": "1.0",
            "__globals__": [],
            "data": self.data.detach().cpu().numpy().T, 
            }
        sio.savemat(path, dict)

