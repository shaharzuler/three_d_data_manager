
import os
import json
from dataclasses import asdict

import numpy as np
import open3d as o3d

from .file_paths import FilePaths
from .data_creators.data_creator_base import DataCreatorBase
from .utils import mesh_utils, voxels_utils , sections_2d_visualization_utils


class Dataset:
    def __init__(self, target_root_dir:str, file_paths:dict=None) -> None:
        self.target_root_dir:str = target_root_dir
        os.makedirs(target_root_dir, exist_ok=True)
        if file_paths is not None:
            self.file_paths = FilePaths(**file_paths)
        else:
            self.file_paths = FilePaths()

    def add_sample(self, data_creator:DataCreatorBase):
        self.file_paths = data_creator.add_sample(self.target_root_dir, self.file_paths, self.__dict__, )
        self.update_properties(data_creator)

    def update_properties(self, data_creator:DataCreatorBase):
        properties = data_creator.get_properties()
        for prop_name, prop_value in properties.items():
            setattr(self, prop_name, prop_value)

    def save_file_paths(self):
        with open(os.path.join(self.target_root_dir, f"file_paths.json"), "a") as f:
            f.write(json.dumps(asdict(self.file_paths)))

    def get_zxy_voxels_mask(self, name:str) -> np.array:
        zxy_voxels_mask_arr = np.load(self.file_paths.zxy_voxels_mask_raw[name])
        return zxy_voxels_mask_arr

    def get_xyz_voxels_mask(self, name:str) -> np.array:
        xyz_voxels_mask_arr = np.load(self.file_paths.xyz_voxels_mask_raw[name])
        return xyz_voxels_mask_arr

    def get_xyz_arr(self, name:str) -> np.array:
        xyz_arr = np.load(self.file_paths.xyz_arr[name])
        return xyz_arr

    def get_zxy_arr(self, name:str) -> np.array:
        xyz_arr = self.get_xyz_arr(name)
        zxy_arr = voxels_utils.xyz_to_zxy(xyz_arr)
        return zxy_arr  

    def get_smooth_voxels_mask(self, name:str) -> np.array:
        xyz_voxels_mask_smooth = np.load(self.file_paths.xyz_voxels_mask_smooth[name])    
        return xyz_voxels_mask_smooth

    def get_mesh(self, name:str) -> np.array:
        mesh = mesh_utils.read_off(self.file_paths.mesh[name])    
        return mesh

    def get_smooth_mesh(self, name:str) -> np.array:
        mesh_smooth = mesh_utils.read_off(self.file_paths.mesh_smooth[name])    
        return mesh_smooth

    def get_convex_mesh(self, name:str) -> np.array:
        mesh_convex = mesh_utils.read_off(self.file_paths.mesh_convex[name])    
        return mesh_convex

    def get_point_cloud(self, name:str) -> np.array:
        point_cloud = np.asarray(o3d.io.read_point_cloud(self.file_paths.point_cloud_from_mesh[name]).points)
        return point_cloud    

    def get_smooth_point_cloud(self, name:str) -> np.array:
        point_cloud = np.asarray(o3d.io.read_point_cloud(self.file_paths.point_cloud_from_mesh_smooth[name]).points)
        return point_cloud  

    def get_convex_point_cloud(self, name:str) -> np.array:
        point_cloud = np.asarray(o3d.io.read_point_cloud(self.file_paths.point_cloud_from_mesh_convex[name]).points)
        return point_cloud 
    
    def get_mesh_lbo_data(self, name:str) -> tuple[np.array, np.array, np.array]:
        lbo_data = np.load(self.file_paths.mesh_lbo_data[name])    
        eigenvectors, eigenvalues, area_weights = lbo_data["eigenvectors"], lbo_data["eigenvalues"], lbo_data["area_weights"], 
        return eigenvectors, eigenvalues, area_weights

    def get_smooth_lbo_data(self, name:str) -> tuple[np.array, np.array, np.array]:
        lbo_data = np.load(self.file_paths.smooth_mesh_lbo_data[name])    
        eigenvectors, eigenvalues, area_weights = lbo_data["eigenvectors"], lbo_data["eigenvalues"], lbo_data["area_weights"], 
        return eigenvectors, eigenvalues, area_weights

    def get_convex_lbo_data(self, name:str) -> tuple[np.array, np.array, np.array]:
        lbo_data = np.load(self.file_paths.convex_mesh_lbo_data[name])    
        eigenvectors, eigenvalues, area_weights = lbo_data["eigenvectors"], lbo_data["eigenvalues"], lbo_data["area_weights"], 
        return eigenvectors, eigenvalues, area_weights
    
    def get_voxelized_mesh(self, name:str) -> np.array:
        voxels_mesh = np.load(self.file_paths.xyz_mesh_voxelized[name])
        return voxels_mesh

    def get_voxelized_smooth_mesh(self, name:str) -> np.array:
        voxels_smooth_mesh = np.load(self.file_paths.xyz_smooth_mesh_voxelized[name])    
        return voxels_smooth_mesh

    def get_voxelized_convex_mesh(self, name:str) -> np.array:
        voxels_convex_mesh = np.load(self.file_paths.xyz_convex_mesh_voxelized[name])    
        return voxels_convex_mesh

    def visualize_existing_data_sections(self, two_d_visualization_data_creator):
        two_d_visualization_data_creator.creation_args.xyz_scan_arr = self.get_xyz_arr(two_d_visualization_data_creator.sample_name)
        two_d_visualization_data_creator.creation_args.masks_data = [
            {
                "name": "raw_mask_sections",
                "arr": self.get_xyz_voxels_mask(two_d_visualization_data_creator.sample_name),
                "color": sections_2d_visualization_utils.colors.YELLOW_RGB
            },
            {
                "name": "smooth_by_voxels_mask_sections",
                "arr": self.get_smooth_voxels_mask(two_d_visualization_data_creator.sample_name),
                "color": sections_2d_visualization_utils.colors.RED_RGB
            },
            {
                "name": "mesh_mask_sections",
                "arr": self.get_voxelized_mesh(two_d_visualization_data_creator.sample_name),
                "color": sections_2d_visualization_utils.colors.PURPLE_RGB
            },
            {
                "name": "smooth_by_lbo_mask_sections",
                "arr":  self.get_voxelized_smooth_mesh(two_d_visualization_data_creator.sample_name),
                "color": sections_2d_visualization_utils.colors.BLUE_RGB
            },
            {
                "name": "convex_mask_sections",
                "arr":  self.get_voxelized_convex_mesh(two_d_visualization_data_creator.sample_name),
                "color": sections_2d_visualization_utils.colors.GREEN_RGB
            },
        ]
        self.add_sample(two_d_visualization_data_creator)

    def visualize_existing_data_3d(self, three_d_visualization_data_creator):
        three_d_visualization_data_creator.creation_args.smooth_mesh_verts, three_d_visualization_data_creator.creation_args.smooth_mesh_faces = self.get_smooth_mesh(three_d_visualization_data_creator.sample_name)
        three_d_visualization_data_creator.creation_args.smooth_mesh_eigenvectors = self.get_mesh_lbo_data(three_d_visualization_data_creator.sample_name)[0]

        self.add_sample(three_d_visualization_data_creator)



        
        



        





