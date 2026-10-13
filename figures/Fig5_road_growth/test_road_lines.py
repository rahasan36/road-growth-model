import os
import sys

import numpy as np
import shapely
from rasterio.features import rasterize
from rasterio.transform import from_origin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import road_lines as RL

CRS = "EPSG:3857"
CELL = 300.0
N = 400
T = from_origin(0, N * CELL, CELL, CELL)


def burn(lines_classes):
    a = np.zeros((N, N), np.uint8)
    for g, k in sorted(lines_classes, key=lambda gk: -gk[1]):
        a[rasterize([(g, 1)], out_shape=(N, N), transform=T) > 0] = k
    return a


def test_new_roads():
    t = np.linspace(0, 1, 400)
    truth = {
        "diagonal 30": (shapely.linestrings(np.c_[5e3 + 25e3 * t, 100e3 + 14.4e3 * t]), 1),
        "diagonal 45": (shapely.linestrings(np.c_[5e3 + 20e3 * t, 70e3 + 20e3 * t]), 2),
        "sine": (shapely.linestrings(np.c_[40e3 + 60e3 * t, 100e3 + 5e3 * np.sin(4 * np.pi * t)]), 3),
        "arc": (shapely.linestrings(np.c_[70e3 + 20e3 * np.cos(np.pi * t), 40e3 + 20e3 * np.sin(np.pi * t)]), 4),
        "parallel a": (shapely.linestrings([[5e3, 50e3], [40e3, 55e3]]), 2),
        "parallel b": (shapely.linestrings([[5e3, 50.9e3], [40e3, 55.9e3]]), 2),
    }
    a = burn(truth.values())
    thick = shapely.linestrings([[60e3, 10e3], [110e3, 20e3]])
    a[rasterize([(thick.buffer(260), 1)], out_shape=(N, N), transform=T) > 0] = 2
    truth["two cells wide"] = (thick, 2)
    g, k, _ = RL.new_road_lines(a, T, CRS, CRS, existing=None, snap=False)
    allg = shapely.union_all(g)
    for name, (line, cls) in truth.items():
        part = shapely.intersection(allg, shapely.buffer(line, 400))
        d_out = shapely.distance(shapely.points(shapely.get_coordinates(shapely.segmentize(part, 30))), line)
        d_back = shapely.distance(shapely.points(shapely.get_coordinates(shapely.segmentize(line, 30))), allg)
        ratio = shapely.length(part) / shapely.length(line)
        print(f"{name:15s} mean {d_out.mean():5.0f} m  max {d_out.max():5.0f} m  98% back {np.percentile(d_back, 98):5.0f} m  length ratio {ratio:.3f}")
        assert d_out.mean() < 0.35 * CELL and d_out.max() < 1.0 * CELL and np.percentile(d_back, 98) <= 0.55 * CELL
        assert 0.95 < ratio < 1.05
        assert set(k[shapely.dwithin(g, line, 150)]) == {cls}
    pa = shapely.union_all(g[shapely.dwithin(g, truth["parallel a"][0], 150)])
    pb = shapely.union_all(g[shapely.dwithin(g, truth["parallel b"][0], 150)])
    assert not shapely.intersects(pa, pb)


def test_junction():
    main = shapely.linestrings([[5e3, 30e3], [45e3, 30e3]])
    branch = shapely.linestrings([[25e3, 30e3], [31e3, 10e3]])
    g, _, _ = RL.new_road_lines(burn([(main, 1), (branch, 3)]), T, CRS, CRS, snap=False)
    assert len(g) == 3
    assert shapely.get_num_geometries(shapely.union_all(shapely.buffer(g, 1.0))) == 1


def test_one_cell_link():
    ex = np.array([shapely.linestrings([[100e3, 89.85e3], [110e3, 89.85e3]]),
                   shapely.linestrings([[100e3, 90.45e3], [110e3, 90.45e3]])])
    a = np.zeros((N, N), np.uint8)
    a[int((N * CELL - 90.15e3) // CELL), int(105.15e3 // CELL)] = 3
    g, k, _ = RL.new_road_lines(a, T, CRS, CRS, existing=ex)
    assert len(g) == 1 and abs(shapely.length(g[0]) - 600) < 1 and list(k) == [3]
    assert all(shapely.distance(g[0], e) < 1 for e in ex)


def test_upgraded():
    lines = np.array([shapely.linestrings([[5e3, 30e3], [45e3, 30e3]]), shapely.linestrings([[5e3, 100e3], [30e3, 114.4e3]])])
    up = np.zeros((N, N), np.uint8)
    up[rasterize([(shapely.linestrings([[10e3, 30e3], [30e3, 30e3]]), 1)], out_shape=(N, N), transform=T) > 0] = 1
    g, kn, ko, st = RL.upgraded_on_lines(lines, np.array([2, 3]), up, T, CRS, CRS)
    km = shapely.length(g).sum() / 1e3
    print(f"upgraded length {km:.2f} km (truth 20 km), cells on lines {st['upgraded_cells_on_lines']}")
    assert abs(km - 20) <= CELL / 1e3 * 2 and set(kn) == {1} and set(ko) == {2} and st["upgraded_cells_on_lines"] == 1.0


if __name__ == "__main__":
    for test in (test_new_roads, test_junction, test_one_cell_link, test_upgraded):
        test()
        print(test.__name__, "passed")
