import os
from distutils.dir_util import copy_tree
import numpy as np
from .file_paths import FilePaths
from .utils import dicom_utils
import shutil
#TODO all creations only if doesnt exist

class DataCreator:
    def __init__(self, source_path, name:str, hirarchy_levels:int) -> None:
        self.source_path:str = source_path
        self.name:str = name
        self.hirarchy_levels:int = hirarchy_levels
        self.default_top_foldername:str = "orig"
        # TODO add automatically creation of 2d image self.create_2d_img = create_2d_img

    def add_sample(self, target_root_dir):
        if self.hirarchy_levels>2:
            self.sample_dir = os.path.join(target_root_dir, self.name, *([self.default_top_foldername]*self.hirarchy_levels))
        else: 
            self.sample_dir = os.path.join(target_root_dir, self.name, self.default_top_foldername)
        os.makedirs(self.sample_dir, exist_ok=True)

    

class DicomDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        # super().__init__(source_path, name, hirarchy_levels)
        self.default_filename = "DICOM"
    
    def add_sample(self, target_root_dir, file_paths:FilePaths):
        super().add_sample(target_root_dir)
        self.dicom_dir = os.path.join(self.sample_dir, self.default_filename)
        os.makedirs(self.dicom_dir, exist_ok=True)
        dicom_source_file_paths = dicom_utils.get_filepaths_by_img_num(self.source_path, int(self.name))
        dicom_target_file_paths = [shutil.copy2(file_path, self.dicom_dir) for file_path in dicom_source_file_paths]
        file_paths.dicom_dir = self.dicom_dir
        file_paths.dicom_file_paths = dicom_target_file_paths

        return file_paths

class XYZArrDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_filename = "xyz_arr"
    
    def get_xyz_arr_from_dicom(self, dicom_dir, target_root_dir):
        xyz_arr = dicom_utils.images_to_3d_arr(dicom_dir, int(self.name))
        return xyz_arr

    def add_sample(self, target_root_dir, file_paths:FilePaths):
        super().add_sample(target_root_dir)
        xyz_arr = self.get_xyz_arr_from_dicom(file_paths.dicom_dir, target_root_dir)#, file_paths)
        self.arr_path = os.path.join(target_root_dir, self.name, self.default_top_foldername, self.default_filename + ".npy")
        np.save(self.arr_path, xyz_arr)
        file_paths.xyz_arr = self.arr_path
        return file_paths



