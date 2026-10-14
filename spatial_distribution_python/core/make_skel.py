"""make_skel.py  -- port of make_skel.m

Identifies branch points and end points of a binary skeleton using the same
hit-or-miss templates as the original (derived from StructuralGT,
https://github.com/drewvecchio/StructuralGT.git).
"""

import numpy as np
from matlab_compat import bwhitmiss


def make_skel(x):
    br = branchedPoints(x)
    ep = endPoints(x)
    br = br - br * ep
    return br, ep


def _a(rows):
    return np.array(rows, dtype=np.uint8)


def branchedPoints(skel):
    xbranch0 = _a([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    xbranch1 = _a([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    tbranch0 = _a([[0, 0, 0], [1, 1, 1], [0, 1, 0]])
    tbranch1 = np.flipud(tbranch0)
    tbranch2 = tbranch0.T.copy()
    tbranch3 = np.fliplr(tbranch2)

    tbranch4 = _a([[1, 0, 1], [0, 1, 0], [1, 0, 0]])
    tbranch5 = np.flipud(tbranch4)
    tbranch6 = np.fliplr(tbranch4)
    tbranch7 = np.fliplr(tbranch5)

    ybranch0 = _a([[1, 0, 1], [0, 1, 0], [0, 1, 0]])
    ybranch1 = np.flipud(ybranch0)
    ybranch2 = ybranch0.T.copy()
    ybranch3 = np.fliplr(ybranch2)

    ybranch4 = _a([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
    ybranch5 = np.flipud(ybranch4)
    ybranch6 = np.fliplr(ybranch4)
    ybranch7 = np.fliplr(ybranch5)

    offbranch0 = _a([[0, 1, 0], [1, 1, 0], [1, 0, 1]])
    offbranch1 = np.flipud(offbranch0)
    offbranch2 = np.fliplr(offbranch0)
    offbranch3 = np.fliplr(offbranch1)
    offbranch4 = offbranch0.T.copy()
    offbranch5 = np.flipud(offbranch4)
    offbranch6 = np.fliplr(offbranch4)
    offbranch7 = np.fliplr(offbranch5)

    clustbranch0 = _a([[0, 1, 1], [0, 1, 1], [1, 0, 0]])
    clustbranch1 = np.flipud(clustbranch0)
    clustbranch2 = np.fliplr(clustbranch0)
    clustbranch3 = np.fliplr(clustbranch1)

    clustbranch4 = _a([[1, 1, 1], [0, 1, 1], [1, 0, 0]])
    clustbranch5 = np.flipud(clustbranch4)
    clustbranch6 = np.fliplr(clustbranch4)
    clustbranch7 = np.fliplr(clustbranch5)

    clustbranch8 = _a([[1, 1, 1], [0, 1, 1], [1, 0, 1]])
    clustbranch9 = np.flipud(clustbranch8)
    clustbranch10 = np.fliplr(clustbranch8)
    clustbranch11 = np.fliplr(clustbranch9)

    crossbranch0 = _a([[1, 0, 0], [1, 1, 1], [0, 1, 0]])
    crossbranch1 = np.flipud(crossbranch0)
    crossbranch2 = np.fliplr(crossbranch0)
    crossbranch3 = np.fliplr(crossbranch1)
    crossbranch4 = crossbranch0.T.copy()
    crossbranch5 = np.flipud(crossbranch4)
    crossbranch6 = np.fliplr(crossbranch4)
    crossbranch7 = np.fliplr(crossbranch5)

    templates = [
        xbranch0, xbranch1, tbranch0, tbranch1, tbranch2, tbranch3, tbranch4,
        tbranch5, tbranch6, tbranch7, ybranch0, ybranch1, ybranch2, ybranch3,
        ybranch4, ybranch5, ybranch6, ybranch7, offbranch0, offbranch1,
        offbranch2, offbranch3, offbranch4, offbranch5, offbranch6, offbranch7,
        clustbranch0, clustbranch1, clustbranch2, clustbranch3, clustbranch4,
        clustbranch5, clustbranch6, clustbranch7, clustbranch8, clustbranch9,
        clustbranch10, clustbranch11, crossbranch0, crossbranch1, crossbranch2,
        crossbranch3, crossbranch4, crossbranch5, crossbranch6, crossbranch7,
    ]

    br = np.zeros_like(np.asarray(skel), dtype=np.int64)
    for t in templates:
        br = br + bwhitmiss(skel, t)  # single-interval form (hit only)
    return (br > 0).astype(np.uint8)


def endPoints(skel):
    endpoint1 = _a([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
    endpoint2 = _a([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
    endpoint3 = _a([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
    endpoint4 = _a([[0, 0, 1], [0, 1, 0], [0, 0, 0]])
    endpoint5 = _a([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
    endpoint6 = _a([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    endpoint7 = _a([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
    endpoint8 = _a([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    endpoint9 = _a([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    endpoint10 = _a([[1, 0, 0], [1, 1, 0], [0, 0, 0]])
    endpoint11 = _a([[1, 1, 0], [0, 1, 0], [0, 0, 0]])
    endpoint12 = _a([[0, 1, 1], [0, 1, 0], [0, 0, 0]])
    endpoint13 = _a([[0, 0, 1], [0, 1, 1], [0, 0, 0]])
    endpoint14 = _a([[0, 0, 0], [0, 1, 1], [0, 0, 1]])
    endpoint15 = _a([[0, 0, 0], [0, 1, 0], [0, 1, 1]])
    endpoint16 = _a([[0, 0, 0], [0, 1, 0], [1, 1, 0]])
    endpoint17 = _a([[0, 0, 0], [1, 1, 0], [1, 0, 0]])

    eps = [endpoint1, endpoint2, endpoint3, endpoint4, endpoint5, endpoint6,
           endpoint7, endpoint8, endpoint9, endpoint10, endpoint11, endpoint12,
           endpoint13, endpoint14, endpoint15, endpoint16, endpoint17]

    ep = np.zeros_like(np.asarray(skel), dtype=np.int64)
    for t in eps:
        ep = ep + bwhitmiss(skel, t, 1 - t)  # two-SE (hit + miss) form
    return (ep > 0).astype(np.uint8)
