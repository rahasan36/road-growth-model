"""matlab_compat.py

MATLAB-compatibility helpers used by the Python port of the road-growth
simulation.  The original code (by A N M Nafiz Abeer) relies heavily on
MATLAB semantics that differ from NumPy:

  * 1-based indexing
  * column-major (Fortran) ordering for find / ind2sub / sub2ind / labelling
  * Image Processing Toolbox functions (bwhitmiss, bwlabeln, bwconncomp,
    label2idx, poly2mask, pdist2 ... 'Smallest')
  * geotiff IO (geotiffread / geotiffwrite)

Every helper here reproduces the MATLAB behaviour as closely as practical so
that the ported algorithm follows the same code paths and, in particular, the
same tie-breaking order (which determines the output).

Coordinate convention used throughout the port: row/column coordinates are
kept 1-based exactly like MATLAB.  When such a coordinate indexes a NumPy
array we subtract 1 at the point of access.
"""

import numpy as np
from scipy import ndimage, signal
from scipy.spatial.distance import cdist
from scipy.io import savemat, loadmat

try:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
    import warnings as _warnings
    _warnings.simplefilter("ignore", NotGeoreferencedWarning)
except Exception:  # pragma: no cover - rasterio optional until IO is used
    rasterio = None


# ---------------------------------------------------------------------------
# Random number generation (seed-gated determinism)
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng()


def set_seed(seed):
    """Set the global RNG.

    If ``seed`` is None the RNG is reseeded from OS entropy (random behaviour,
    matching MATLAB's ``randomized = 1`` with no fixed seed).  If an integer is
    given the runs become reproducible.
    """
    global _RNG
    if seed is None:
        _RNG = np.random.default_rng()
    else:
        _RNG = np.random.default_rng(int(seed))


def randperm0(n):
    """Return a 0-based permutation of ``range(n)`` (like MATLAB randperm(n)-1).

    Used to permute Python (0-based) index arrays.
    """
    return _RNG.permutation(int(n))


# ---------------------------------------------------------------------------
# Scalar math helpers (degree-based trig + MATLAB rounding/wrapping)
# ---------------------------------------------------------------------------
def matlab_round(x):
    """MATLAB round: round half away from zero (NumPy rounds half to even)."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def sind(d):
    return np.sin(np.radians(np.asarray(d, dtype=float)))


def cosd(d):
    return np.cos(np.radians(np.asarray(d, dtype=float)))


def atan2d(y, x):
    return np.degrees(np.arctan2(np.asarray(y, dtype=float),
                                 np.asarray(x, dtype=float)))


def wrapTo360(a):
    """Replicates MATLAB wrapTo360: wrap to [0, 360], positive multiples -> 360."""
    a = np.asarray(a, dtype=float)
    scalar = a.ndim == 0
    a = np.atleast_1d(a)
    positive = a > 0
    out = np.mod(a, 360.0)
    out[(out == 0) & positive] = 360.0
    return float(out[0]) if scalar else out


def colon(a, b, step=1.0):
    """Replicates MATLAB ``a:step:b`` (inclusive, fractional start allowed)."""
    a = float(a)
    b = float(b)
    step = float(step)
    if step == 0:
        return np.array([])
    n = int(np.floor((b - a) / step + 1e-10))
    if n < 0:
        return np.array([])
    return a + step * np.arange(0, n + 1)


# ---------------------------------------------------------------------------
# Indexing helpers (1-based, column-major)
# ---------------------------------------------------------------------------
def find(A):
    """MATLAB ``[r,c,v] = find(A)`` returning 1-based row, col, value arrays in
    column-major order over the non-zero entries of ``A``.
    """
    A = np.asarray(A)
    mask = A != 0
    # nonzero of the transpose iterates the original array column-major
    cols, rows = np.nonzero(mask.T)
    vals = A[rows, cols]
    return rows + 1, cols + 1, vals


def find_linear(A):
    """MATLAB ``find(A)`` -> 1-based column-major linear indices of non-zeros."""
    A = np.asarray(A)
    R = A.shape[0]
    rows, cols, _ = find(A)
    return (cols - 1) * R + rows


def sub2ind(sz, r, c):
    """1-based column-major linear index (vectorised)."""
    R = sz[0]
    r = np.asarray(r)
    c = np.asarray(c)
    return (c - 1) * R + r


def ind2sub(sz, ind):
    """Inverse of :func:`sub2ind`; returns (row, col) 1-based arrays."""
    R = sz[0]
    ind = np.asarray(ind)
    ind0 = ind - 1
    r = ind0 % R + 1
    c = ind0 // R + 1
    return r, c


# ---------------------------------------------------------------------------
# Connected components / labelling (column-major label order, like MATLAB)
# ---------------------------------------------------------------------------
def _structure(conn):
    if conn == 8:
        return np.ones((3, 3), dtype=bool)
    elif conn == 4:
        return ndimage.generate_binary_structure(2, 1)
    raise ValueError("conn must be 4 or 8")


def bwlabeln(BW, conn=8):
    """Like MATLAB ``bwlabeln(BW, conn)`` with labels numbered in column-major
    order of first occurrence (matching MATLAB's labelling order).
    """
    BW = np.asarray(BW) != 0
    lbl, n = ndimage.label(BW, structure=_structure(conn))
    if n == 0:
        return lbl.astype(np.int64), 0
    R = lbl.shape[0]
    rows, cols = np.nonzero(lbl)
    labs = lbl[rows, cols]
    lin = cols.astype(np.int64) * R + rows  # 0-based column-major
    min_per = np.full(n + 1, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(min_per, labs, lin)
    order = np.argsort(min_per[1:], kind="stable")  # old labels sorted by first pixel
    remap = np.zeros(n + 1, dtype=np.int64)
    remap[order + 1] = np.arange(1, n + 1)
    return remap[lbl], n


def label2idx(L):
    """Like MATLAB ``label2idx``: list (len = max label) of 1-based column-major
    linear indices for each label, each sorted ascending.
    """
    L = np.asarray(L)
    n = int(L.max()) if L.size else 0
    R = L.shape[0]
    rows, cols = np.nonzero(L)
    labs = L[rows, cols]
    lin = cols.astype(np.int64) * R + rows + 1  # 1-based column-major
    out = [np.array([], dtype=np.int64) for _ in range(n)]
    for k in range(1, n + 1):
        out[k - 1] = np.sort(lin[labs == k])
    return out


def bwconncomp(BW, conn=8):
    """Like MATLAB ``bwconncomp``; returns ``PixelIdxList`` as a list of arrays
    of 1-based column-major linear indices, components in column-major order.
    """
    L, _ = bwlabeln(BW, conn)
    return label2idx(L)


# ---------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------
def bwhitmiss(BW, se1, se2=None):
    """MATLAB ``bwhitmiss``.

    Two forms:
      * ``bwhitmiss(BW, interval)`` where ``interval`` holds 1 (hit) / 0
        (don't care).  Equivalent to erosion by the 1-positions.
      * ``bwhitmiss(BW, se1, se2)`` where ``se1`` marks pixels that must be
        foreground and ``se2`` marks pixels that must be background.
    """
    BW = np.asarray(BW) != 0
    if se2 is None:
        s1 = np.asarray(se1) == 1
        s2 = np.zeros_like(s1)
    else:
        s1 = np.asarray(se1) != 0
        s2 = np.asarray(se2) != 0
    return ndimage.binary_hit_or_miss(BW, structure1=s1,
                                      structure2=s2).astype(np.uint8)


# ---------------------------------------------------------------------------
# poly2mask
# ---------------------------------------------------------------------------
def poly2mask(c_coords, r_coords, R, C):
    """MATLAB ``poly2mask(x, y, m, n)`` with x=cols(c), y=rows(r), m=R, n=C.

    Uses skimage.draw.polygon (center-in-polygon test).  Minor boundary
    differences from MATLAB are possible; see README.
    """
    from skimage.draw import polygon
    rr, cc = polygon(np.asarray(r_coords, dtype=float) - 1,
                     np.asarray(c_coords, dtype=float) - 1, shape=(R, C))
    mask = np.zeros((R, C), dtype=bool)
    mask[rr, cc] = True
    return mask


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------
def pdist2_smallest(X, Y, metric="euclidean", k=1):
    """MATLAB ``pdist2(X, Y, metric, 'Smallest', 1)`` -> for each row of Y the
    smallest distance to any row of X (1 x size(Y,1)).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    m = "sqeuclidean" if metric in ("squaredeuclidean", "sqeuclidean") else metric
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return np.array([])
    D = cdist(X, Y, metric=m)
    if k == 1:
        return D.min(axis=0)
    idx = np.argsort(D, axis=0)[:k, :]
    return np.take_along_axis(D, idx, axis=0)


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------
def conv2_same(A, kernel):
    """MATLAB ``conv2(A, kernel, 'same')`` (zero padded)."""
    return signal.convolve2d(np.asarray(A, dtype=float),
                             np.asarray(kernel, dtype=float),
                             mode="same", boundary="fill", fillvalue=0)


# ---------------------------------------------------------------------------
# GeoTIFF IO
# ---------------------------------------------------------------------------
def geotiffread(path):
    """Like MATLAB ``[A, R] = geotiffread(path)``.

    Returns the first band as a NumPy array and a reference dict carrying the
    rasterio profile (transform + crs) needed to write outputs back out.
    """
    if rasterio is None:
        raise ImportError("rasterio is required for geotiffread/geotiffwrite")
    with rasterio.open(path) as ds:
        A = ds.read(1)
        R = {"transform": ds.transform, "crs": ds.crs, "profile": dict(ds.profile)}
    return A, R


def geotiffwrite(path, A, R):
    """Like MATLAB ``geotiffwrite(path, single(A), R, ...)``.

    Writes a single-band float32 GeoTIFF preserving the source geotransform.
    The original MATLAB code overrides the projection GeoKeys to user-defined
    (GTModelTypeGeoKey=1, ProjectedCSTypeGeoKey=32767); here we instead
    preserve the input CRS/transform.  Pixel values and georeferencing
    transform are identical; see README for the CRS-tag note.
    """
    if rasterio is None:
        raise ImportError("rasterio is required for geotiffread/geotiffwrite")
    A = np.asarray(A, dtype=np.float32)
    prof = dict(R["profile"])
    prof.update(driver="GTiff", count=1, dtype="float32",
                height=A.shape[0], width=A.shape[1])
    prof["transform"] = R["transform"]
    prof["crs"] = R["crs"]
    if "nodata" in prof and prof["nodata"] is not None:
        # keep as-is; values are written verbatim
        pass
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(A, 1)


# ---------------------------------------------------------------------------
# .mat IO (kept MATLAB-compatible so the two-stage pipeline interoperates)
# ---------------------------------------------------------------------------
def save_mat(path, **vars):
    savemat(path, vars)


def load_mat(path):
    return loadmat(path)
