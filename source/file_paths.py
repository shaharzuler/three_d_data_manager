from typing import List
from dataclasses import dataclass


@dataclass
class FilePaths:
    def __init__(self) -> None:
        self.dicom_dir:str = None
        self.dicom_file_paths:List = None
        self.xyz_arr:str = None
        # self.zxy_arr:str = None
        self.two_d_x:str = None
        self.two_d_y:str = None
        self.two_d_z:str = None
        self.scan_sections:str = None
        self.section_contours:str = None

        #TODO add section names
        

        self.zxy_voxels_mask_raw:str = None
        self.xyz_voxels_mask_raw:str = None
        self.xyz_voxels_mask_smooth:str = None

        self.mesh:str = None
        self.mesh_smooth:str = None
        self.lbo_data:str = None
        self.mesh_convex:str = None
        self.pcd_raw:str = None

        self.mesh_voxelized:str = None
        self.smooth_mesh_voxelized:str = None
        self.convex_mesh_voxelized:str = None
