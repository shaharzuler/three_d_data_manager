from typing import List
from dataclasses import dataclass

@dataclass
class FilePaths:
    def __init__(self) -> None:
        self.dicom_dir:str = None
        self.dicom_file_paths:List = None
        self.xyz_arr:str = None

        self.zxy_voxels_mask_raw:str = None

        self.xyz_voxels_mask_raw:str = None
        self.xyz_voxels_mask_smooth:str = None

        self.mesh:str = None
        self.mesh_lbo_data:str = None
        self.mesh_dataset:str = None
        self.xyz_mesh_voxelized:str = None

        self.mesh_smooth:str = None
        self.mesh_smooth_lbo_data:str = None
        self.mesh_smooth_dataset:str = None
        self.xyz_smooth_mesh_voxelized:str = None

        self.mesh_convex:str = None
        self.mesh_convex_lbo_data:str = None
        self.mesh_convex_dataset:str = None
        self.xyz_convex_mesh_voxelized:str = None

        self.scan_sections:str = None
        self.section_contours:str = None
        self.n0_raw_mask_sections :str = None
        self.n1_smooth_by_voxels_mask_sections:str = None
        self.n2_mesh_mask_sections:str = None
        self.n3_smooth_by_lbo_mask_sections:str = None
        self.n4_convex_mask_sections:str = None

        self.lbo_visualization:dict = None
        self.clear_mesh_visualization:dict = None
