import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Fig5_road_growth"))
import make_fig5_road_growth as F

if __name__ == "__main__":
    F.main("sfig6", HERE)
