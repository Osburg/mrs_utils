#%%
from utils.transform import FourierTransform, InverseFourierTransform, OneHotEncodingTransform
import torch
import numpy as np
from scipy.signal import find_peaks
from torch.fft import fft, fftshift

import matplotlib.pyplot as plt

def test_fourier_transform():
    dt = 0.001
    N = 10000

    t = torch.linspace(-N/2*dt, N/2*dt, N, dtype=torch.complex64)
    s = (10*np.cos(2*np.pi*5*t)+5*np.cos(2*np.pi*40*t))*np.exp(-np.pi*t**2)
    f = torch.abs(FourierTransform()(s))
    peaks, _ = find_peaks(f, height=10)
    assert np.allclose(torch.linspace(-1/(2*dt), 1/(2*dt), N)[peaks], [-40,-5,5,40], atol=1/(N*dt))

def test_inverse_fourier_transform():
    dt = 0.001
    N = 10000

    t = torch.linspace(-N/2*dt, N/2*dt, N, dtype=torch.complex64)
    s = (10*np.cos(2*np.pi*5*t)+5*np.cos(2*np.pi*40*t))*np.exp(-np.pi*t**2)
    f = fftshift(fft(s, norm="ortho"))
    s_rec = InverseFourierTransform()(f)

    assert np.allclose(s_rec, s, atol=1e-5)


def test_one_hot_transform():
    n_classes = 10
    ohe = OneHotEncodingTransform(n_classes)
    x = torch.tensor([0,1,2,3,4,5,6,7,8,9])
    
    for x in torch.tensor([0,5,2,2,1,7,3,6,8,9]):
        y = ohe(x)
        assert torch.all(y.sum() == 1)
        assert torch.all(y.argmax() == x)
