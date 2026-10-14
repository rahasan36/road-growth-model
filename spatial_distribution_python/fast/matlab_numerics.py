"""Functions that reproduce MATLAB numeric behaviour where NumPy differs."""
import numpy as np
from scipy import ndimage

def matlab_class_div(A, w):
    """double(A*w/max(max(A))) evaluated in A's own class, as MATLAB does.
    Integer rasters: saturating integer ops with round-half-away division.
    single rasters: float32 arithmetic, then cast to double."""
    A = np.asarray(A)
    if np.issubdtype(A.dtype, np.integer):
        info = np.iinfo(A.dtype)
        Aw = np.clip(np.sign(A * float(w)) * np.floor(np.abs(A * float(w)) + 0.5), info.min, info.max)
        mx = float(A.max())
        q = Aw / mx
        q = np.sign(q) * np.floor(np.abs(q) + 0.5)
        return np.clip(q, info.min, info.max).astype(np.float64)
    if A.dtype == np.float32:
        A32 = A * np.float32(w)
        return (A32 / np.float32(A.max())).astype(np.float64)
    A = A.astype(np.float64)
    return A * w / np.max(A)

def bwhitmiss_matlab(BW, se1, se2=None):
    """bwhitmiss = imerode(BW,SE1) & imerode(~BW,SE2); imerode pads binary images with 1."""
    BW = np.asarray(BW) != 0
    if se2 is None:
        iv = np.asarray(se1); s1 = iv == 1; s2 = iv == -1
    else:
        s1 = np.asarray(se1) != 0; s2 = np.asarray(se2) != 0
    hit = ndimage.binary_erosion(BW, structure=s1, border_value=1) if s1.any() else np.ones_like(BW)
    miss = ndimage.binary_erosion(~BW, structure=s2, border_value=1) if s2.any() else np.ones_like(BW)
    return (hit & miss).astype(np.uint8)

def apply():
    import matlab_compat, make_skel
    matlab_compat.bwhitmiss = bwhitmiss_matlab
    make_skel.bwhitmiss = bwhitmiss_matlab
