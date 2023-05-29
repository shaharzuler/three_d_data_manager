import os
from distutils.dir_util import copy_tree
from typing import Dict
import shutil
from dataclasses import asdict

import numpy as np
import scipy.ndimage
import open3d as o3d

# from three_d_data_manager.source.utils import 
# from three_d_data_manager.source.utils import 
# from three_d_data_manager.source.utils import visualization_utils #visualize_grid_of_lbo
# from three_d_data_manager.source.utils import 
#   
from .file_paths import FilePaths
from .utils import dicom_utils, os_utils, mesh_utils, voxels_utils, LBO_utils, voxels_mesh_conversions_utils



#TODO all creations only if doesnt exist
#TODO call visualization args for 2d dicom. visualize_grid_of_lbos etc

class DataCreator:
    def __init__(self, source_path, name:str, hirarchy_levels:int) -> None:
        self.source_path:str = source_path
        self.name:str = name
        self.hirarchy_levels:int = hirarchy_levels
        self.default_top_foldername:str = "orig"
        # TODO add automatically creation of 2d image self.create_2d_img = create_2d_img

    def add_sample(self, target_root_dir:str, creation_args, dataset_attrs:Dict[str,str]):
        if self.hirarchy_levels>2:
            self.sample_dir = os.path.join(target_root_dir, self.name, *([self.default_top_foldername]*self.hirarchy_levels))
        else: 
            self.sample_dir = os.path.join(target_root_dir, self.name, self.default_top_foldername)
        os.makedirs(self.sample_dir, exist_ok=True)

    def add_sample_from_file(self, file, target_root_dir:str, file_paths:FilePaths, creation_args, dataset_attrs:Dict[str,str]):
        raise NotImplementedError

    def get_properties(self) -> Dict[str, any]:
        return {}


    




class DicomDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        # super().__init__(source_path, name, hirarchy_levels)
        self.default_filename = "DICOM"
    
    def add_sample(self, target_root_dir, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None):
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
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

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None):
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
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
    
    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None):
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        self.arr_path = os.path.join(target_root_dir, self.name, self.default_top_foldername, self.default_filename + ".npy")
        if "voxel_size" in dataset_attrs:
            voxel_size_zxy = dataset_attrs["voxel_size"]
            arr = scipy.ndimage.zoom(arr, voxel_size_zxy)
        # shutil.copy2(self.source_path, self.arr_path)
        np.save(self.arr_path, arr)
        
        # shutil.copy2(self.source_path, self.arr_path)
        file_paths.zxy_voxels_mask_raw = self.arr_path
        return file_paths

    def add_sample_from_file(self, file:np.array, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        target_folder_path = os.path.join(target_root_dir, self.name, self.default_top_foldername)
        self.arr_path = os.path.join(target_folder_path, self.default_filename + ".npy")
        np.save(self.arr_path, file)
        file_paths.xyz_voxels_mask_raw = self.arr_path

        return file_paths

class ZXYVoxelsMaskDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_filename = "zxy_voxels_mask_raw"
    
    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None):
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        self.arr_path = os.path.join(target_root_dir, self.name, self.default_top_foldername, self.default_filename + ".npy")
        arr = np.load(self.source_path)
        if voxels_utils.zxy_to_xyz(arr).shape != dataset_attrs["shape"]:
            if "voxel_size" in dataset_attrs:
                voxel_size_zxy = dataset_attrs["voxel_size"][2], dataset_attrs["voxel_size"][0], dataset_attrs["voxel_size"][1]
                arr = scipy.ndimage.zoom(arr, voxel_size_zxy)
        # shutil.copy2(self.source_path, self.arr_path)
        np.save(self.arr_path, arr.astype(bool))
        file_paths.zxy_voxels_mask_raw = self.arr_path
        return file_paths

class SmoothVoxelsMaskDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int, file=None) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_filename = "xyz_voxels_mask_smooth"
        self.file = file
    
    def add_sample_from_file(self, file:np.array, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        target_folder_path = os.path.join(target_root_dir, self.name, self.default_top_foldername)
        self.arr_path = os.path.join(target_folder_path, self.default_filename + ".npy")
        np.save(self.arr_path, file)
        file_paths.xyz_voxels_mask_smooth = self.arr_path

        os_utils.write_config_file(target_folder_path, self.default_filename, asdict(creation_args)) 
        return file_paths

class MeshDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int, file=None) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_filename = "mesh"
        self.file = file
    
    def add_sample_from_file(self, file:np.array, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        target_folder_path = os.path.join(target_root_dir, self.name, self.default_top_foldername)
        self.mesh_path = os.path.join(target_folder_path, self.default_filename + ".off")
        verts, faces = file
        mesh_utils.write_off(self.mesh_path, verts, faces)
        file_paths.mesh = self.mesh_path

        # os_utils.write_config_file(target_folder_path, self.default_filename, asdict(args)) #TODO add voxels start index
        return file_paths

class SmoothLBOMeshDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_filename = "smooth_mesh"
    

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None):
        super().add_sample(target_root_dir, creation_args, dataset_attrs)

        verts, faces = mesh_utils.read_off(file_paths.mesh)
        smooth_verts, faces = self.smooth_with_lbo(creation_args, verts, faces)

        target_folder_path = os.path.join(target_root_dir, self.name, self.default_top_foldername)
        self.smooth_mesh_path = os.path.join(target_folder_path, self.default_filename + ".off")
        mesh_utils.write_off(self.smooth_mesh_path, smooth_verts, faces)
        self.lbo_data_path =  os.path.join(target_folder_path, "lbo_data.npz")
        np.savez(self.lbo_data_path, eigenvectors=self.eigenvectors, eigenvalues=self.eigenvalues, area_weights=self.area_weights)
        
        file_paths.mesh_smooth = self.smooth_mesh_path
        file_paths.lbo_data = self.lbo_data_path

        os_utils.write_config_file(target_folder_path, self.default_filename, asdict(creation_args)) 

        return file_paths

   
    def smooth_with_lbo(self, lbo_creation_args, verts, faces):
        LBO = LBO_utils.LBOcalc(k=lbo_creation_args.num_LBOs, use_torch=lbo_creation_args.use_torch, is_point_cloud=lbo_creation_args.is_point_cloud)
        self.eigenvectors, self.eigenvalues, self.area_weights = LBO.get_LBOs(verts, faces)
        # the following is to reduce dim in the LBO space (verts*eigenvects*eigenvects^-1) :
        projected = np.dot(verts.transpose(), self.eigenvectors[0,:,:])
        eigenvects_pinv = np.linalg.pinv(self.eigenvectors[0,:,:])
        smooth_verts = np.dot(projected, eigenvects_pinv).transpose() 
        return smooth_verts, faces




class ConvexMeshDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_filename = "convex_mesh"
    

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None):
        super().add_sample(target_root_dir, creation_args, dataset_attrs)

        verts, faces = mesh_utils.read_off(file_paths.mesh_smooth)
        convex_verts, convex_faces = self.convexify(verts)

        target_folder_path = os.path.join(target_root_dir, self.name, self.default_top_foldername) #TODO as 1 func for base class? prepare paths first for check if file exists?
        self.convex_mesh_path = os.path.join(target_folder_path, self.default_filename + ".off")
        mesh_utils.write_off(self.convex_mesh_path, convex_verts, convex_faces)
        
        file_paths.mesh_convex = self.convex_mesh_path

        return file_paths

    def convexify(self, vertices):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        convex_hull, orig_pcd_point_indices = pcd.compute_convex_hull()
        verts_convex_hull = np.asarray(convex_hull.vertices).astype(np.float32)
        faces_convex_hull = np.asarray(convex_hull.triangles).astype(np.int64)
        return verts_convex_hull, faces_convex_hull

class VoxelizedMeshDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_filename = "_voxelized"

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None):
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        mesh_filename = creation_args.mesh_path.split("/")[-1].split(".off")[0]
        voxelized = voxels_mesh_conversions_utils.Mesh2VoxelsConvertor(creation_args.mesh_path, dataset_attrs["shape"]).padded_voxelized

        self.voxelized_mesh_path = os.path.join(target_root_dir, self.name, self.default_top_foldername, mesh_filename + self.default_filename + ".npy")
        np.save(self.voxelized_mesh_path, voxelized)
        setattr(file_paths, mesh_filename + self.default_filename, self.voxelized_mesh_path)

        return file_paths

    

