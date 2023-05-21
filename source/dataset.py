import os
import numpy as np

from .file_paths import FilePaths
from .data_creators import DataCreator
from .utils.voxels_utils import xyz_to_zxy_arr
#TODO maybe user decides which format gets saved and which is created on the fly.



class Dataset:
    def __init__(self, target_root_dir:str) -> None:
        self.target_root_dir:str = target_root_dir
        os.makedirs(target_root_dir, exist_ok=True)
        self.file_paths = FilePaths()

    def add_sample(self, data_creator:DataCreator):
        self.file_paths = data_creator.add_sample(self.target_root_dir, self.file_paths)

    def get_xyz_arr(self):
        xyz_arr = np.load(self.file_paths.xyz_arr)
        return xyz_arr

    def get_zxy_arr(self):
        xyz_arr = self.get_xyz_arr()
        zxy_arr = xyz_to_zxy_arr(xyz_arr)
        return zxy_arr      

    


        





