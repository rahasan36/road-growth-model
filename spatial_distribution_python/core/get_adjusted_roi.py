"""get_adjusted_roi.py  -- port of get_adjusted_ROI.m

region of interest centered around (r0, c0).
Returns 1-based ROI row/column coordinate vectors (column-major flatten,
matching MATLAB ``(:)``).
"""

import numpy as np


def get_adjusted_ROI(r0, c0, r1, r2, c1, c2):
    temp_r = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]], dtype=float)
    temp_c = temp_r.T.copy()

    if (r0 - r1) * (r0 - r2) * (c0 - c1) * (c0 - c2) != 0:
        ROI_r = temp_r
        ROI_c = temp_c
    else:
        if r0 == r1:
            if c0 == c1:
                ROI_r = temp_r[1:, 1:]
                ROI_c = temp_c[1:, 1:]
            elif c0 == c2:
                ROI_r = temp_r[1:, 0:2]
                ROI_c = temp_c[1:, 0:2]
            else:
                ROI_r = temp_r[1:, :]
                ROI_c = temp_c[1:, :]
        elif r0 == r2:
            if c0 == c1:
                ROI_r = temp_r[0:2, 1:]
                ROI_c = temp_c[0:2, 1:]
            elif c0 == c2:
                ROI_r = temp_r[0:2, 0:2]
                ROI_c = temp_c[0:2, 0:2]
            else:
                ROI_r = temp_r[0:2, :]
                ROI_c = temp_c[0:2, :]
        else:
            if c0 == c1:
                ROI_r = temp_r[:, 1:]
                ROI_c = temp_c[:, 1:]
            else:
                ROI_r = temp_r[:, 0:2]
                ROI_c = temp_c[:, 0:2]

    ROI_r = ROI_r.flatten(order="F") + r0
    ROI_c = ROI_c.flatten(order="F") + c0
    return ROI_r.astype(np.int64), ROI_c.astype(np.int64)
