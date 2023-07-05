from dataclasses import dataclass
import inspect

@dataclass
class FilePaths:

    dicom_dir: dict[str, str] = None
    dicom_file_paths:dict[str, list] = None
    xyz_arr: dict[str, str] = None

    zxy_voxels_mask_raw: dict[str, str] = None

    xyz_voxels_mask_raw: dict[str, str] = None
    xyz_voxels_mask_smooth: dict[str, str] = None

    mesh: dict[str, str] = None
    mesh_lbo_data: dict[str, str] = None
    mesh_dataset: dict[str, str] = None
    point_cloud_from_mesh: dict[str, str] = None
    point_cloud_from_mesh_lbo_data: dict[str, str] = None
    point_cloud_from_mesh_dataset: dict[str, str] = None
    xyz_mesh_voxelized: dict[str, str] = None

    mesh_smooth: dict[str, str] = None
    mesh_smooth_lbo_data: dict[str, str] = None
    mesh_smooth_dataset: dict[str, str] = None
    point_cloud_from_mesh_smooth: dict[str, str] = None
    point_cloud_from_mesh_smooth_lbo_data: dict[str, str] = None
    point_cloud_from_mesh_smooth_dataset: dict[str, str] = None
    xyz_smooth_mesh_voxelized: dict[str, str] = None

    mesh_convex: dict[str, str] = None
    mesh_convex_lbo_data: dict[str, str] = None
    mesh_convex_dataset: dict[str, str] = None
    point_cloud_from_mesh_convex: dict[str, str] = None
    point_cloud_from_mesh_convex_lbo_data: dict[str, str] = None
    point_cloud_from_mesh_convex_dataset: dict[str, str] = None
    xyz_convex_mesh_voxelized: dict[str, str] = None


    scan_sections: dict[str, str] = None
    section_contours: dict[str, str] = None
    n0_raw_mask_sections: dict[str, str] = None
    n1_smooth_by_voxels_mask_sections: dict[str, str] = None
    n2_mesh_mask_sections: dict[str, str] = None
    n3_smooth_by_lbo_mask_sections: dict[str, str] = None
    n4_convex_mask_sections: dict[str, str] = None

    lbo_visualization: dict[str, str] = None
    clear_mesh_visualization: dict[str, str] = None

    def add_path(self, path_name:str, name:str, val:any):
        if getattr(self, path_name) is None:
            attr = {name: val}
        else:
            attr = getattr(self, path_name)
            attr[name] = val
        setattr(self, path_name, attr)

        
