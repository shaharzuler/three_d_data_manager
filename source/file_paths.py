#todo make dataclass
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
        self.two_d:str = None

        self.zxy_voxels_mask_raw:str = None
        self.xyz_voxels_mask_raw:str = None
        self.xyz_voxels_mask_smooth:str = None

        self.mesh:str = None
        self.mesh_smooth:str = None
        self.lbo_data:str = None
        self.mesh_convex:str = None
        self.pcd_raw:str = None
