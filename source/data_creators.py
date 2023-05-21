import os
from distutils.dir_util import copy_tree
import numpy as np
from dataset import FilePaths
from utils.dicom_utils import images_to_3d_arr


class DataCreator:
    def __init__(self, source_path, name:str, hirarchy_levels:int) -> None:
        self.source_path:str = source_path
        self.name = name
        self.hirarchy_levels = hirarchy_levels
        self.default_top_foldername = "orig"
        # TODO add automatically creation of 2d image self.create_2d_img = create_2d_img

    def add_sample(self, target_root_dir):
        self.sample_dir = os.path.join(target_root_dir, self.name, *([self.default_top_foldername]*self.hirarchy_levels))
        os.makedirs(self.sample_dir)

    

class DicomDataCreator:
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super.__init__(source_path, name, hirarchy_levels)
        self.default_filename = "DICOM"
    
    def add_sample(self, target_root_dir, file_paths:FilePaths):
        super.add_sample(target_root_dir)
        self.dicom_dir = os.path.join(self.sample_dir, self.default_filename)
        os.makedirs(self.dicom_dir)
        copy_tree(self.source_path, self.dicom_dir)
        file_paths.dicom = self.dicom_dir

        return file_paths

class XYZArrDataCreator:
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super.__init__(source_path, name, hirarchy_levels)
        self.default_filename = "xyz_arr"
    
    def add_sample_from_dicom(self, ):
        xyz_arr = images_to_3d_arr(self.file_paths.dicom)
        self.arr_path = os.path.join(self.source_path, self.default_filename)
        np.save(self.arr_path, xyz_arr)

    def add_sample(self, target_root_dir, file_paths:FilePaths):
        super.add_sample(target_root_dir)
        file_paths = self.add_sample_from_dicom(target_root_dir, file_paths)
        file_paths.xyz_arr = self.arr_path
        return file_paths

# class ZXYArrDataCreator:
#     def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
#         super.__init__(source_path, name, hirarchy_levels)
#         self.default_filename = "zxy_arr"
    
#     def add_sample_from_dicom(self):
#         xyz_arr = images_to_3d_arr(self.file_paths.dicom)
#         self.arr_path = os.path.join(self.source_path, self.default_filename)
#         np.save(self.arr_path, xyz_arr)

#     def add_sample_from_xyz_arr(self, xyz_arr):
#         zxy_arr = xyz_to_zxy_arr(xyz_arr)
    

#     def add_sample(self, target_root_dir, file_paths:FilePaths):
#         super.add_sample(target_root_dir)
#         if self.source_path.endswith(".npy"):
#             xyz_arr = np.load(self.source_path)
#             self.add_sample_from_xyz_arr(xyz_arr)
#         else:
#             file_paths = self.add_sample_from_dicom(target_root_dir, file_paths)
#         file_paths.xyz_arr = self.arr_path
#         return file_paths

