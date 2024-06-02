import numpy as np
from typing import List
from typing_extensions import Self
import scipy.io as sio
import re
from datetime import datetime

class Basisset():
    """Class to store metabolite basis sets."""

    def __init__(self, 
                 fids: np.ndarray, 
                 metabolite_names: List[str], 
                 conjugate_basis: bool = False, 
                 dwelltime: float = None
                ) -> None:
        """Initializes the Basisset object.

        Args:
            fids (np.ndarray): 2D numpy array The data to be stored in the container. 
                The samples should be stored along the first dimension.
            metabolite_names (List[str]): List of metabolite names.
            conjugate_basis (bool): Optional. If True, the basis set is conjugated.
            dwelltime (float): Optional. The dwell time of the FID signals.
        """
        if not len(metabolite_names) == fids.shape[1]:
            raise ValueError("The number of metabolite names should be equal to the number of FID signals.")
        self.fids: np.ndarray = np.array(fids, dtype=np.complex64)
        self.spectra = np.array([
            np.fft.fftshift(
                np.fft.fft(self.fids, axis=0)[:,i]) for i in range(self.fids.shape[1])
            ], dtype=np.complex64).T
        self.metabolite_names: List[str] = metabolite_names

        # TODO: Optional attributes
        self.dwelltime = dwelltime
        self.conjugate_basis = conjugate_basis
        
    def __len__(self) -> int:
        return self.fids.shape[1]
        
    def __getitem__(self, idx: int) -> np.ndarray:
        return self.fids[:,idx]
    
    def __setitem__(self, idx: int, value: np.ndarray) -> None:
        self.fids[:,idx] = value
        self.spectra[:,idx] = np.fft.fftshift(np.fft.fft(value))
    
    def __iter__(self):
        return self.fids.T.__iter__()
    
    def __next__(self):
        return self.fids.T.__next__()
    
    def get_name_from_index(self, idx: int) -> str:
        """Returns the metabolite name corresponding to the index."""
        return self.metabolite_names[idx]

    def conjugate(self):
        """Conjugates the basis set."""
        self.conjugate_basis = not self.conjugate_basis
        self.fids = np.conj(self.fids)
        # based on fourier trafo properties
        self.spectra = np.flipud(np.conj(self.spectra)) 
    
    @classmethod
    def from_matlab(cls, path: str, metabolite_names: List[str] = None) -> Self:
        """
        Args:
            path (str): Path to the .mat file containing the fid signals and the metabolite names.
            metabolite_names (List[str]): Optional. List of metabolite names. If None is
                provided, the names will be taken from the field "metabolite_names" of the input file. 
            
        Returns:
            A Basisset object.
        """
        fids = sio.loadmat(path).get('data')
        if metabolite_names is None:
            metabolite_names = sio.loadmat(path).get('metabolite_names')
        return cls(fids, metabolite_names)
    
    @classmethod
    def from_npz(cls, path: str, metabolite_names: List[str]) -> Self:
        """
        Args:
            path (str): Path to the npz file containing the fid .
            metabolite_names (List[str]): List of metabolite names.
        
        Returns:
            A Basisset object.
        """
        data = np.load(path)
        data = data["data"].T
        return cls(data, metabolite_names, False, None)
    
    @classmethod
    def from_RAW(cls, path_list: List[str], metabolite_names: List[str] = None) -> Self:
        """
        Args:
            path_list (List[str]): List of paths to the .raw files.
            metabolite_names List[str]: Optional. List of metabolite names.
                if no names are provided, the names will be taken from the field
                "ID" of the input files.

        Returns:
            A Basisset object.
        """
        if metabolite_names is not None:
            assert len(metabolite_names) == len(path_list)

        out = {}
        for path in path_list:
            metabolite = io_readlcmraw(path)
            out.update(metabolite)
        
        metabolite_names = list(out.keys())
        fids = np.array([out[metab] for metab in metabolite_names]).T
        conjugate_basis = False

        return cls(fids, metabolite_names, conjugate_basis)
    
    @classmethod
    def from_BASIS(cls, path: str) -> Self:
        """
        Args: 
            path (str): Path to the .basis file containing the basis.
        """
        out = io_readlcmraw_basis(path, conjugate=True)
        
        metabolite_names = list(out.keys())
        fids = np.array([out[metab]['fids'] for metab in metabolite_names]).T
        conjugate_basis = True
        # assuming the same dwelltime for all metabolites
        dwelltime = out[metabolite_names[0]]['dwelltime']

        return cls(fids, metabolite_names, conjugate_basis, dwelltime)

    def normalize(self):
        """Normalizes the basis set."""
        NotImplementedError


def io_readlcmraw(filename: str) -> dict:
    """
    Read a LCModel .raw file and extract the FID signal from it.

    Parameters:
    filename (str): The path to the .raw file.

    Returns:
    dict: A dictionary containing the metabolite name as key and the FID signal as value.
    """
    out = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        
        if "ID" in line and not "NMID" in line:
            metabolite_name = line.split(sep="=")[1]
            metabolite_name = metabolite_name.split(".RAW")[0] 
        
        if "$END" in line:
            # Read the FID
            real = []
            imag = []
            while line.strip() and i < len(lines)-1:
                i+=1
                line = lines[i]
                vals = line.split()
                real += [float(vals[0])]
                imag += [float(vals[1])]

            fid = np.array(real) + 1j * np.array(imag)
            out[metabolite_name] = fid

        i+=1

    
    return out


########################################################################################
##### The following code is translated from MATLAB to Python from the repo        ######
##### plotLCM by schorschinho (https://github.com/schorschinho/plotLCMBasis)      ######
##### using chatGPT.                                                              ######
########################################################################################

def get_num_from_string(s):
    s = re.sub(r'[;=]', ' ', s)
    s = re.sub(r'[^\d.\-eE]', ' ', s)
    s = re.sub(r'(?i)e(?![+-])', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def remove_white_spaces(s):
    return re.sub(r'\s+', '', s)

def lcmodel_rng(dix):
    a = 16807.0
    b15 = 32768.0
    b16 = 65536.0
    p = 2147483647.0
    
    xhi = dix / b16
    xhi = xhi - (xhi % 1.0)
    xalo = (dix - xhi * b16) * a
    leftlo = xalo / b16
    leftlo = leftlo - (leftlo % 1.0)
    fhi = xhi * a + leftlo
    k = fhi / b15
    k = k - (k % 1.0)
    dix = (((xalo - leftlo * b16) - p) + (fhi - k * b15) * b16) + k
    if dix < 0.0:
        dix += p
    randomresult = dix * 4.656612875e-10
    return randomresult, dix

def io_readlcmraw_basis(filename, conjugate=True):
    """
    Read .BASIS file and extract the FID signals and metabolite names.

    Args:
        filename (str): The path to the .basis file.
        conjugate (bool, optional): Whether to conjugate the FID signals. Defaults to True.

    Returns:
        dict: A dictionary containing the extracted information from the .basis file.

    Raises:
        FileNotFoundError: If the specified file does not exist.

    """
    out = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    linewidth = None
    hzpppm = None
    te = None
    dwelltime = None
    Bo = None
    linewidth = None
    spectralwidth = None
    centerFreq = None
    metabName = None
    
    i = 0
    while i < len(lines):
        line = lines[i]

        if linewidth is None:
            while 'FWHMBA' not in line:
                i += 1
                line = lines[i]    
            linewidth = float(get_num_from_string(line))
        
        if hzpppm is None:
            while 'HZPPPM' not in line:
                i += 1
                line = lines[i]
            hzpppm = float(get_num_from_string(line))
            Bo = hzpppm / 42.577
            linewidth = linewidth * hzpppm

        if te is None:
            while 'ECHOT' not in line:
                i += 1
                line = lines[i]
            te = float(get_num_from_string(line))

        if dwelltime is None:
            while 'BADELT' not in line:
                i += 1
                line = lines[i]
            dwelltime = float(get_num_from_string(line))
            spectralwidth = 1 / dwelltime

        if centerFreq is None:
            while 'PPMSEP' not in line:
                i += 1
                line = lines[i]
                if 'METABO' in line and not 'METABO_' in line:
                    break
            if 'PPMSEP' in line:
                centerFreq = float(get_num_from_string(line))
            else:
                centerFreq = []
        
        if metabName is None:
            while not('METABO' in line and not 'METABO_' in line):
                i += 1
                line = lines[i]
            metabName = re.search(r"METABO\s*=\s*['\"]?([-_+A-Za-z0-9]+)['\"]?", line).group(1)
            
        
        if '$END' in line:
            i += 1
            line = lines[i]
            RF = []
            while not any(x in line for x in ['$NMUSED', '$BASIS']) and line.strip():
                RF += [float(val) for val in line.split()]
                i += 1
                if i < len(lines):
                    line = lines[i]
                else:
                    break
            
            specs = np.array(RF[0::2]) + 1j * np.array(RF[1::2])
            
            if dwelltime < 0:
                dix = 1499
                for rr in range(len(specs)):
                    randomresult, dix = lcmodel_rng(dix)
                    specs[rr] = -specs[rr] * np.exp(-20 * randomresult + 10)
            
            if conjugate:
                # osburg: here the conjugation property of the fourier transform is used?
                specs = np.flipud(np.fft.fftshift(np.conj(specs)))
            else:
                specs = np.fft.fftshift(specs)
            
            vectorsize = len(specs)
            sz = (vectorsize, 1)
            if vectorsize % 2 == 0:
                fids = np.fft.ifft(np.fft.ifftshift(specs))
            else:
                fids = np.fft.ifft(np.roll(np.fft.ifftshift(specs), 1))
            
            f = np.linspace((-1+1/sz[0])*spectralwidth/2 , (1-1/sz[0])*spectralwidth/2, vectorsize)
            ppm = f / (Bo * 42.577) + 4.68
            t = np.arange(dwelltime, vectorsize * dwelltime + dwelltime, dwelltime)
            txfrq = hzpppm * 1e6
            metabName = remove_white_spaces(metabName)
            if metabName == '-CrCH2':
                metabName = 'CrCH2'
            if metabName == '2HG':
                metabName = 'bHG'
            
            out[metabName] = {
                'fids': fids,
                'specs': specs,
                'sz': (vectorsize, 1, 1, 1),
                'n': vectorsize,
                'spectralwidth': abs(spectralwidth),
                'Bo': Bo,
                'te': te,
                'tr': [],
                'dwelltime': abs(1 / spectralwidth),
                'linewidth': linewidth,
                'ppm': ppm,
                't': t,
                'txfrq': txfrq,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'seq': '',
                'sim': '',
                'dims': {
                    't': 1,
                    'coils': 0,
                    'averages': 0,
                    'subSpecs': 0,
                    'extras': 0
                },
                'averages': 1,
                'flags': {
                    'writtentostruct': 1,
                    'gotparams': 1,
                    'leftshifted': 1,
                    'filtered': 0,
                    'zeropadded': 0,
                    'freqcorrected': 0,
                    'phasecorrected': 0,
                    'averaged': 1,
                    'addedrcvrs': 1,
                    'subtracted': 1,
                    'writtentotext': 1,
                    'downsampled': 0,
                    'isFourSteps': 0
                }
            }

            centerFreq = None
            metabName = None

        i += 1
    
    return out