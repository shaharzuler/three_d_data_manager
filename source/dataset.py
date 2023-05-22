import os
import numpy as np
from three_d_data_manager.source.utils.voxels_utils import xyz_to_zxy

from .file_paths import FilePaths
from .data_creators import DataCreator
# from .utils.voxels_utils import xyz_to_zxy
#TODO maybe user decides which format gets saved and which is created on the fly.



class Dataset:
    def __init__(self, target_root_dir:str) -> None:
        self.target_root_dir:str = target_root_dir
        os.makedirs(target_root_dir, exist_ok=True)
        self.file_paths = FilePaths()

    def add_sample(self, data_creator:DataCreator,):
        self.file_paths = data_creator.add_sample(self.target_root_dir, self.file_paths,  self.__dict__)
        self.update_properties(data_creator)

    def add_sample_from_file(self, data_creator:DataCreator, args=None):
        self.file_paths = data_creator.add_sample_from_file(data_creator.file, self.target_root_dir, self.file_paths, args, self.__dict__)

    def update_properties(self, data_creator:DataCreator):
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
        zxy_arr = xyz_to_zxy(xyz_arr)
        return zxy_arr      

    


        





