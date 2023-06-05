
import os
from typing import Tuple

import numpy as np

from .file_paths import FilePaths
from .data_creators import DataCreatorBase
from .utils import mesh_utils, voxels_utils , sections_2d_visualization_utils

# TODO sample name in getters and fileaths (!)
# TODO maybe user desides which format gets saved and which is created on the fly. (*)





class Dataset:
    def __init__(self, target_root_dir:str) -> None:
        self.target_root_dir:str = target_root_dir
        os.makedirs(target_root_dir, exist_ok=True)
        self.file_paths = FilePaths()

    def add_sample(self, data_creator:DataCreatorBase, creation_args=None):
        self.file_paths = data_creator.add_sample(self.target_root_dir, self.file_paths,  creation_args, self.__dict__, )
        self.update_properties(data_creator)

    def add_sample_from_file(self, data_creator:DataCreatorBase, creation_args=None):
        self.file_paths = data_creator.add_sample_from_file(data_creator.file, self.target_root_dir, self.file_paths, creation_args, self.__dict__)

    def update_properties(self, data_creator:DataCreatorBase):
        properties = data_creator.get_properties()
        for prop_name, prop_value in properties.items():
            setattr(self, prop_name, prop_value)

    def get_zxy_voxels_mask(self) -> np.array:
        zxy_voxels_mask_arr = np.load(self.file_paths.zxy_voxels_mask_raw)
        return zxy_voxels_mask_arr

    def get_xyz_voxels_mask(self) -> np.array:
        xyz_voxels_mask_arr = np.load(self.file_paths.xyz_voxels_mask_raw)
        return xyz_voxels_mask_arr

    def get_xyz_arr(self) -> np.array:
        xyz_arr = np.load(self.file_paths.xyz_arr)
        return xyz_arr

    def get_zxy_arr(self) -> np.array:
        xyz_arr = self.get_xyz_arr()
        zxy_arr = voxels_utils.xyz_to_zxy(xyz_arr)
        return zxy_arr  

    def get_smooth_voxels_mask(self) -> np.array:
        xyz_voxels_mask_smooth = np.load(self.file_paths.xyz_voxels_mask_smooth)    
        return xyz_voxels_mask_smooth

    def get_mesh(self) -> np.array:
        mesh = mesh_utils.read_off(self.file_paths.mesh)    
        return mesh

    def get_smooth_mesh(self) -> np.array:
        mesh_smooth = mesh_utils.read_off(self.file_paths.mesh_smooth)    
        return mesh_smooth

    def get_convex_mesh(self) -> np.array:
        mesh_convex = mesh_utils.read_off(self.file_paths.mesh_convex)    
        return mesh_convex
    
    def get_mesh_lbo_data(self) -> Tuple[np.array, np.array, np.array]:
        lbo_data = np.load(self.file_paths.mesh_lbo_data)    
        eigenvectors, eigenvalues, area_weights = lbo_data["eigenvectors"], lbo_data["eigenvalues"], lbo_data["area_weights"], 
        return eigenvectors, eigenvalues, area_weights

    def get_smooth_lbo_data(self) -> Tuple[np.array, np.array, np.array]:
        lbo_data = np.load(self.file_paths.smooth_mesh_lbo_data)    
        eigenvectors, eigenvalues, area_weights = lbo_data["eigenvectors"], lbo_data["eigenvalues"], lbo_data["area_weights"], 
        return eigenvectors, eigenvalues, area_weights

    def get_convex_lbo_data(self) -> Tuple[np.array, np.array, np.array]:
        lbo_data = np.load(self.file_paths.convex_mesh_lbo_data)    
        eigenvectors, eigenvalues, area_weights = lbo_data["eigenvectors"], lbo_data["eigenvalues"], lbo_data["area_weights"], 
        return eigenvectors, eigenvalues, area_weights
    
    def get_voxelized_mesh(self) -> np.array:
        voxels_mesh = np.load(self.file_paths.xyz_mesh_voxelized)
        return voxels_mesh

    def get_voxelized_smooth_mesh(self) -> np.array:
        voxels_smooth_mesh = np.load(self.file_paths.xyz_smooth_mesh_voxelized)    
        return voxels_smooth_mesh

    def get_voxelized_convex_mesh(self) -> np.array:
        voxels_convex_mesh = np.load(self.file_paths.xyz_convex_mesh_voxelized)    
        return voxels_convex_mesh

    def visualize_existing_data_sections(self, two_d_visualisation_data_creator, two_d_visualisation_args):
        two_d_visualisation_args.xyz_scan_arr = self.get_xyz_arr()
        two_d_visualisation_args.masks_data = [
            {
                "name": "raw_mask_sections",
                "arr": self.get_xyz_voxels_mask(),
                "color": sections_2d_visualization_utils.colors.YELLOW_RGB
            },
            {
                "name": "smooth_by_voxels_mask_sections",
                "arr": self.get_smooth_voxels_mask(),
                "color": sections_2d_visualization_utils.colors.RED_RGB
            },
            {
                "name": "mesh_mask_sections",
                "arr": self.get_voxelized_mesh(),
                "color": sections_2d_visualization_utils.colors.PURPLE_RGB
            },
            {
                "name": "smooth_by_lbo_mask_sections",
                "arr":  self.get_voxelized_smooth_mesh(),
                "color": sections_2d_visualization_utils.colors.BLUE_RGB
            },
            {
                "name": "convex_mask_sections",
                "arr":  self.get_voxelized_convex_mesh(),
                "color": sections_2d_visualization_utils.colors.GREEN_RGB
            },
        ]
        self.add_sample(two_d_visualisation_data_creator, two_d_visualisation_args)

    def visualize_existing_data_3d(self, three_d_visualisation_data_creator, three_d_visualisation_args):
        three_d_visualisation_args.smooth_mesh_verts, three_d_visualisation_args.smooth_mesh_faces = self.get_smooth_mesh()
        three_d_visualisation_args.smooth_mesh_eigenvectors = self.get_mesh_lbo_data()[0]

        self.add_sample(three_d_visualisation_data_creator, three_d_visualisation_args)



        
        



        





