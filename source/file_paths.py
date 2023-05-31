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
        #TODO h5
        
        self.zxy_voxels_mask_raw:str = None
        self.xyz_voxels_mask_raw:str = None
        self.xyz_voxels_mask_smooth:str = None

        self.mesh:str = None
        self.mesh_smooth:str = None
        self.mesh_lbo_data:str = None
        self.smooth_mesh_lbo_data:str = None
        self.convex_mesh_lbo_data:str = None

        self.mesh_convex:str = None
        self.pcd_raw:str = None

        self.xyz_mesh_voxelized:str = None
        self.xyz_smooth_mesh_voxelized:str = None
        self.xyz_convex_mesh_voxelized:str = None

        self.lbo_visualization:str = None
        self.clear_mesh_visualization:str = None
