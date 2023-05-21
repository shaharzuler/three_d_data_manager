import os
import numpy as np

from data_creators import DataCreator
from three_d_data_manager.source.utils.voxels_utils import xyz_to_zxy_arr


#todo make dataclass
class FilePaths:
    def __init__(self) -> None:
        self.dicom = None
        self.xyz_arr = None
        self.zxy_arr = None
        self.two_d_x = None
        self.two_d_y = None
        self.two_d_z = None
        self.two_d = None
        self.mesh_raw = None
        self.pcd_raw = None


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

    


        





