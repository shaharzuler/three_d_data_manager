from typing import List
from dataclasses import dataclass
from webbrowser import get

@dataclass
class FilePaths:
    def __init__(self) -> None:
        self.dicom_dir: dict[str, str] = None
        self.dicom_file_paths:dict[str, List] = None
        self.xyz_arr: dict[str, str] = None

        self.zxy_voxels_mask_raw: dict[str, str] = None

        self.xyz_voxels_mask_raw: dict[str, str] = None
        self.xyz_voxels_mask_smooth: dict[str, str] = None

        self.mesh: dict[str, str] = None
        self.mesh_lbo_data: dict[str, str] = None
        self.mesh_dataset: dict[str, str] = None
        self.xyz_mesh_voxelized: dict[str, str] = None

        self.mesh_smooth: dict[str, str] = None
        self.mesh_smooth_lbo_data: dict[str, str] = None
        self.mesh_smooth_dataset: dict[str, str] = None
        self.xyz_smooth_mesh_voxelized: dict[str, str] = None

        self.mesh_convex: dict[str, str] = None
        self.mesh_convex_lbo_data: dict[str, str] = None
        self.mesh_convex_dataset: dict[str, str] = None
        self.xyz_convex_mesh_voxelized: dict[str, str] = None

        self.scan_sections: dict[str, str] = None
        self.section_contours: dict[str, str] = None
        self.n0_raw_mask_sections: dict[str, str] = None
        self.n1_smooth_by_voxels_mask_sections: dict[str, str] = None
        self.n2_mesh_mask_sections: dict[str, str] = None
        self.n3_smooth_by_lbo_mask_sections: dict[str, str] = None
        self.n4_convex_mask_sections: dict[str, str] = None

        self.lbo_visualization: dict[str, str] = None
        self.clear_mesh_visualization: dict[str, str] = None

    def add_path(self, path_name:str, name:str, val:any):
        if getattr(self, path_name) is None:
            attr = {name: val}
        else:
            attr = getattr(self, path_name)
            attr[name] = val
        setattr(self, path_name, attr)
        
