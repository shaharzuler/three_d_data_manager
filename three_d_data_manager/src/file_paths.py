from dataclasses import dataclass
from typing import Dict

@dataclass
class FilePaths:

    dicom_dir: Dict[str, str] = None
    dicom_file_paths:Dict[str, list] = None
    xyz_arr: Dict[str, str] = None

    zxy_voxels_mask_raw: Dict[str, str] = None
    zxy_voxels_extra_mask_raw: Dict[str, str] = None

    xyz_voxels_mask_raw: Dict[str, str] = None
    xyz_voxels_extra_mask_raw: Dict[str, str] = None
    xyz_voxels_mask_smooth: Dict[str, str] = None
    xyz_voxels_extra_mask_smooth: Dict[str, str] = None

    mesh: Dict[str, str] = None
    mesh_lbo_data: Dict[str, str] = None
    mesh_dataset: Dict[str, str] = None
    point_cloud_from_mesh: Dict[str, str] = None
    point_cloud_from_mesh_lbo_data: Dict[str, str] = None
    point_cloud_from_mesh_dataset: Dict[str, str] = None
    xyz_mesh_voxelized: Dict[str, str] = None

    mesh_smooth: Dict[str, str] = None
    mesh_smooth_lbo_data: Dict[str, str] = None
    mesh_smooth_dataset: Dict[str, str] = None
    point_cloud_from_mesh_smooth: Dict[str, str] = None
    point_cloud_from_mesh_smooth_lbo_data: Dict[str, str] = None
    point_cloud_from_mesh_smooth_dataset: Dict[str, str] = None
    xyz_smooth_mesh_voxelized: Dict[str, str] = None

    mesh_convex: Dict[str, str] = None
    mesh_convex_lbo_data: Dict[str, str] = None
    mesh_convex_dataset: Dict[str, str] = None
    point_cloud_from_mesh_convex: Dict[str, str] = None
    point_cloud_from_mesh_convex_lbo_data: Dict[str, str] = None
    point_cloud_from_mesh_convex_dataset: Dict[str, str] = None
    xyz_convex_mesh_voxelized: Dict[str, str] = None

    scan_sections: Dict[str, str] = None
    section_contours: Dict[str, str] = None
    n0_raw_mask_sections: Dict[str, str] = None
    n1_smooth_by_voxels_mask_sections: Dict[str, str] = None
    n2_mesh_mask_sections: Dict[str, str] = None
    n3_smooth_by_lbo_mask_sections: Dict[str, str] = None
    n4_convex_mask_sections: Dict[str, str] = None

    lbo_visualization: Dict[str, str] = None
    clear_mesh_visualization: Dict[str, str] = None

    def add_path(self, path_name:str, name:str, val:any):
        if getattr(self, path_name) is None:
            attr = {name: val}
        else:
            attr = getattr(self, path_name)
            attr[name] = val
        setattr(self, path_name, attr)

        
