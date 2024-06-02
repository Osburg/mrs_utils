import torch
from torch import Tensor
from torch.fft import fft, fftshift, ifft, ifftshift
import numpy as np

class FourierTransform:
    def __call__(self, x: Tensor) -> Tensor:
        """FFT of an input signal x"""
        return fftshift(fft(x, norm="ortho"))

class InverseFourierTransform:
    """Inverse FFT of an input signal x"""
    def __call__(self, x: Tensor) -> Tensor:
        return ifft(ifftshift(x), norm="ortho")

class OneHotEncodingTransform:
    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        
    def __call__(self, x: Tensor) -> Tensor:
        if not np.allclose(x.shape, [self.n_classes]):
            raise ValueError("Input tensor must be one-dimensional and of lenght n_classes")
        return torch.zeros(self.n_classes, dtype=torch.float).scatter_(0, index=torch.tensor(x), value=1)
