from dataclasses import dataclass

import numpy as np

@dataclass
class TwoDVisualizationCreationArgs:
    masks_data: dict = None
    xyz_scan_arr: np.array = None
