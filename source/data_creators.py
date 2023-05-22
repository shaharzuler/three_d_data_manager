import os
from distutils.dir_util import copy_tree
from typing import Dict
import numpy as np
from three_d_data_manager.source.utils import voxels_utils
from .file_paths import FilePaths
from .utils import dicom_utils, os_utils
import shutil
from dataclasses import asdict
import scipy.ndimage

#TODO all creations only if doesnt exist

class DataCreator:
    def __init__(self, source_path, name:str, hirarchy_levels:int) -> None:
        self.source_path:str = source_path
        self.name:str = name
        self.hirarchy_levels:int = hirarchy_levels
        self.default_top_foldername:str = "orig"
        # TODO add automatically creation of 2d image self.create_2d_img = create_2d_img

    def add_sample(self, target_root_dir:str, dataset_attrs:Dict[str,str]):
        if self.hirarchy_levels>2:
            self.sample_dir = os.path.join(target_root_dir, self.name, *([self.default_top_foldername]*self.hirarchy_levels))
        else: 
            self.sample_dir = os.path.join(target_root_dir, self.name, self.default_top_foldername)
        os.makedirs(self.sample_dir, exist_ok=True)

    def add_sample_from_file(self, file, target_root_dir:str, file_paths:FilePaths, args, dataset_attrs:Dict[str,str]):
        raise NotImplementedError

    def get_properties(self) -> Dict[str, any]:
        return {}


    




class DicomDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        # super().__init__(source_path, name, hirarchy_levels)
        self.default_filename = "DICOM"
    
    def add_sample(self, target_root_dir, file_paths:FilePaths, dataset_attrs:Dict[str,str]):
        super().add_sample(target_root_dir, dataset_attrs)
        self.dicom_dir = os.path.join(self.sample_dir, self.default_filename)
        os.makedirs(self.dicom_dir, exist_ok=True)
        dicom_source_file_paths = dicom_utils.get_filepaths_by_img_num(self.source_path, int(self.name))
        dicom_target_file_paths = [shutil.copy2(file_path, self.dicom_dir) for file_path in dicom_source_file_paths]
        file_paths.dicom_dir = self.dicom_dir
        file_paths.dicom_file_paths = dicom_target_file_paths
        self.dicom_dir = self.dicom_dir
        self.dicom_file_paths = dicom_target_file_paths

        return file_paths

    def get_properties(self) -> Dict[str, any]:
        slice_path = self.dicom_file_paths[0]
        voxel_size = dicom_utils.get_voxel_size(slice_path)
        properties = {
            "voxel_size": voxel_size
        }
        return properties

class XYZArrDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_filename = "xyz_arr_raw"
    
    def get_xyz_arr_from_dicom(self, dicom_dir):
        self.xyz_arr = dicom_utils.images_to_3d_arr(dicom_dir, int(self.name))
        return self.xyz_arr

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, dataset_attrs:Dict[str,str]):
        super().add_sample(target_root_dir, dataset_attrs)
        xyz_arr = self.get_xyz_arr_from_dicom(file_paths.dicom_dir)
        self.arr_path = os.path.join(target_root_dir, self.name, self.default_top_foldername, self.default_filename + ".npy")
        np.save(self.arr_path, xyz_arr)
        file_paths.xyz_arr = self.arr_path
        return file_paths

    def get_properties(self):
        properties = {
            "shape": self.xyz_arr.shape
        }
        return properties

class XYZVoxelsMaskDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int, file=None) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_filename = "xyz_voxels_mask_raw"
        self.file = file
    
    def add_sample(self, target_root_dir:str, file_paths:FilePaths, dataset_attrs:Dict[str,str]):
        super().add_sample(target_root_dir, dataset_attrs)
        self.arr_path = os.path.join(target_root_dir, self.name, self.default_top_foldername, self.default_filename + ".npy")
        if "voxel_size" in dataset_attrs:
            voxel_size_zxy = dataset_attrs["voxel_size"]
            arr = scipy.ndimage.zoom(arr, voxel_size_zxy)
        # shutil.copy2(self.source_path, self.arr_path)
        np.save(self.arr_path, arr)
        
        # shutil.copy2(self.source_path, self.arr_path)
        file_paths.zxy_voxels_mask_raw = self.arr_path
        return file_paths

    def add_sample_from_file(self, file:np.array, target_root_dir:str, file_paths:FilePaths, args, dataset_attrs:Dict[str,str]) -> FilePaths:
        
        super().add_sample(target_root_dir, dataset_attrs)
        target_folder_path = os.path.join(target_root_dir, self.name, self.default_top_foldername)
        self.arr_path = os.path.join(target_folder_path, self.default_filename + ".npy")
        np.save(self.arr_path, file)
        file_paths.xyz_voxels_mask_raw = self.arr_path

        return file_paths

class ZXYVoxelsMaskDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_filename = "zxy_voxels_mask_raw"
    
    def add_sample(self, target_root_dir:str, file_paths:FilePaths, dataset_attrs:Dict[str,any]):
        super().add_sample(target_root_dir, dataset_attrs)
        self.arr_path = os.path.join(target_root_dir, self.name, self.default_top_foldername, self.default_filename + ".npy")
        arr = np.load(self.source_path)
        if voxels_utils.zxy_to_xyz(arr).shape != dataset_attrs["shape"]:
            if "voxel_size" in dataset_attrs:
                voxel_size_zxy = dataset_attrs["voxel_size"][2], dataset_attrs["voxel_size"][0], dataset_attrs["voxel_size"][1]
                arr = scipy.ndimage.zoom(arr, voxel_size_zxy)
        # shutil.copy2(self.source_path, self.arr_path)
        np.save(self.arr_path, arr)
        file_paths.zxy_voxels_mask_raw = self.arr_path
        return file_paths

class SmoothVoxelsMaskDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int, file=None) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_filename = "xyz_voxels_mask_smooth"
        self.file = file
    
    def add_sample_from_file(self, file:np.array, target_root_dir:str, file_paths:FilePaths, args, dataset_attrs:Dict[str,str]) -> FilePaths:
        super().add_sample(target_root_dir, dataset_attrs)
        target_folder_path = os.path.join(target_root_dir, self.name, self.default_top_foldername)
        self.arr_path = os.path.join(target_folder_path, self.default_filename + ".npy")
        np.save(self.arr_path, file)
        file_paths.xyz_voxels_mask_smooth = self.arr_path

        os_utils.write_config_file(target_folder_path, self.default_filename, asdict(args)) 
        return file_paths



