from dataclasses import dataclass
import numpy as np


@dataclass
class VoxelSmoothingArgs:
    opening_footprint_radius: int
    fill_holes_Area_threshold: int
    closing_to_opening_ratio: float

@dataclass
class MeshSmoothingArgs:
    marching_cubes_step_size: int
    # convexify_method = None #TODO
    
@dataclass
class LBOArgs:
    num_LBOs: int
    is_point_cloud: bool
    mesh_path: str
    orig_mesh_name: str
    use_torch: bool = True # use_torch=False function is buggy

@dataclass
class SmoothMeshCreationArgs:
    lbos_path: str

@dataclass
class VoxelizingArgs:
    mesh_path: str

@dataclass
class TwoDVisArgs:
    masks_data: dict = None
    xyz_scan_arr: np.array = None

@dataclass
class ThreeDVisArgs:
    max_smooth_lbo_mesh_visualization: int

@dataclass
class H5DatasetArgs:
    orig_name: str
    override: bool = False

    
