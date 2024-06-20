from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor
from torch.fft import fft, fftshift, ifft, ifftshift


class Transform(ABC):
    """Base class for all transforms. Must implement a __call__ method."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        return NotImplemented


class FourierTransform(Transform):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Tensor) -> Tensor:
        """FFT of an input signal x"""
        return fftshift(fft(x))


class InverseFourierTransform(Transform):
    """Inverse FFT of an input signal x"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Tensor) -> Tensor:
        return ifft(ifftshift(x))


class OneHotEncodingTransform(Transform):
    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes

    def __call__(self, x: Tensor) -> Tensor:
        for x_ in x.flatten():
            if x_ not in range(self.n_classes):
                raise ValueError(
                    f"Class {x_} is not in the range [0, {self.n_classes})"
                )
        return torch.zeros((x.shape[0], self.n_classes), dtype=torch.float).scatter_(
            -1, index=torch.tensor(x), value=1
        )


class AugmentationTransform(Transform):

    def __init__(
        self,
        frequency_shift: float,
        phase_shift: float,
        damping: float,
        noise_level: float,
        time: Tensor,
        domain: str = "time",
    ) -> None:
        """Class that applied a random frequency shift, phase shift, damping and noise addition to an input signal.

        Args:
            frequency_shift (float): Maximum frequency shift in Hz
            phase_shift (float): Maximum phase shift in radians
            damping (float): Maximum damping factor
            noise_level (float): Maximum noise level
            time (Tensor): Time tensor
            domain (str): Domain of the input signals to be transformed. Can be 'time' or 'frequency'.
        """
        super().__init__()

        self.frequency_shift = frequency_shift
        self.phase_shift = phase_shift
        self.damping = damping
        self.noise_level = noise_level
        self.time = time
        if domain not in ["time", "frequency"]:
            raise ValueError("Domain must be 'time' or 'frequency'")
        self.domain = domain

    def __call__(self, x: Tensor) -> Tensor:
        frequency_shift = self.frequency_shift * (
            torch.rand(x.shape[0]) - 0.5
        ).unsqueeze(1)
        phase_shift = (
            self.phase_shift * np.pi * (torch.rand(x.shape[0]) - 0.5).unsqueeze(1)
        )
        damping = self.damping * torch.rand(x.shape[0]).unsqueeze(1)

        if self.domain == "frequency":
            x = ifft(ifftshift(x, dim=-1), dim=-1)

        x *= torch.exp(1j * (phase_shift + 2 * np.pi * frequency_shift * self.time))
        x *= torch.exp(-damping * self.time)
        x += self.noise_level * (torch.randn(x.shape) + 1j * torch.randn(x.shape))

        if self.domain == "frequency":
            x = fftshift(fft(x, dim=-1), dim=-1)

        return x


class IdentityTransform(Transform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Tensor) -> Tensor:
        return x
