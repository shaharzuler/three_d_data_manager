
#TODO all creations only if doesnt exist
#TODO aplit to files

import os
from typing import Dict
import shutil
from dataclasses import asdict

import numpy as np
import scipy.ndimage
import open3d as o3d
import h5py

from .file_paths import FilePaths
from .utils import dicom_utils, os_utils, mesh_utils, voxels_utils, LBO_utils, voxels_mesh_conversions_utils, sections_2d_visualization_utils, mesh_3d_visualization_utils



class DataCreator:
    def __init__(self, source_path, name:str, hirarchy_levels:int) -> None:
        self.source_path: str = source_path
        self.name: str = name
        self.hirarchy_levels: int = hirarchy_levels
        self.default_top_foldername: str = "orig"

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
        self.default_dirname = "voxels"
        self.default_filename = "xyz_arr_raw"
    
    def get_xyz_arr_from_dicom(self, dicom_dir):
        self.xyz_arr = dicom_utils.images_to_3d_arr(dicom_dir, int(self.name))
        return self.xyz_arr

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None):
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        self.voxels_dir = os.path.join(self.sample_dir, self.default_dirname)
        os.makedirs(self.voxels_dir, exist_ok=True)

        xyz_arr = self.get_xyz_arr_from_dicom(file_paths.dicom_dir)
        self.arr_path = os.path.join(self.voxels_dir, self.default_filename + ".npy")
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
        self.default_dirname = "voxels"
        self.default_filename = "xyz_voxels_mask_raw"
        self.file = file
    
    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None):
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        self.voxels_dir = os.path.join(self.sample_dir, self.default_dirname)
        os.makedirs(self.voxels_dir, exist_ok=True)

        self.arr_path = os.path.join(self.voxels_dir, self.default_filename + ".npy")
        if "voxel_size" in dataset_attrs:
            voxel_size_zxy = dataset_attrs["voxel_size"]
            arr = scipy.ndimage.zoom(arr, voxel_size_zxy)
        np.save(self.arr_path, arr)
        
        file_paths.zxy_voxels_mask_raw = self.arr_path
        return file_paths

    def add_sample_from_file(self, file:np.array, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        self.voxels_dir = os.path.join(self.sample_dir, self.default_dirname)
        os.makedirs(self.voxels_dir, exist_ok=True)

        self.arr_path = os.path.join(self.voxels_dir, self.default_filename + ".npy")
        np.save(self.arr_path, file)
        file_paths.xyz_voxels_mask_raw = self.arr_path

        return file_paths

class ZXYVoxelsMaskDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "voxels"
        self.default_filename = "zxy_voxels_mask_raw"
    
    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None):
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        self.voxels_dir = os.path.join(self.sample_dir, self.default_dirname)
        os.makedirs(self.voxels_dir, exist_ok=True)

        self.arr_path = os.path.join(self.voxels_dir, self.default_filename + ".npy")
        arr = np.load(self.source_path)
        if voxels_utils.zxy_to_xyz(arr).shape != dataset_attrs["shape"]:
            if "voxel_size" in dataset_attrs:
                voxel_size_zxy = dataset_attrs["voxel_size"][2], dataset_attrs["voxel_size"][0], dataset_attrs["voxel_size"][1]
                arr = scipy.ndimage.zoom(arr, voxel_size_zxy)
        np.save(self.arr_path, arr.astype(bool))
        file_paths.zxy_voxels_mask_raw = self.arr_path
        return file_paths

class SmoothVoxelsMaskDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int, file=None) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "voxels"
        self.default_filename = "xyz_voxels_mask_smooth"
        self.file = file
    
    def add_sample_from_file(self, file:np.array, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        self.voxels_dir = os.path.join(self.sample_dir, self.default_dirname)
        os.makedirs(self.voxels_dir, exist_ok=True)

        self.arr_path = os.path.join(self.voxels_dir, self.default_filename + ".npy")
        np.save(self.arr_path, file)
        file_paths.xyz_voxels_mask_smooth = self.arr_path

        os_utils.write_config_file(self.voxels_dir, self.default_filename, asdict(creation_args)) 
        return file_paths

class MeshDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int, file=None) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "meshes"
        self.default_filename = "mesh" 
        self.file = file
    
    def add_sample_from_file(self, file:np.array, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        self.mesh_dir = os.path.join(self.sample_dir, self.default_dirname)
        os.makedirs(self.mesh_dir, exist_ok=True)
        self.mesh_path = os.path.join(self.mesh_dir, self.default_filename + ".off")
        verts, faces = file
        mesh_utils.write_off(self.mesh_path, verts, faces)
        file_paths.mesh = self.mesh_path

        # os_utils.write_config_file(target_folder_path, self.default_filename, asdict(args)) 
        return file_paths

class SmoothLBOMeshDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "meshes"
        self.default_filename = "smooth_mesh"     

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None):
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        self.mesh_dir = os.path.join(self.sample_dir, self.default_dirname)
        os.makedirs(self.mesh_dir, exist_ok=True)

        verts, faces = mesh_utils.read_off(file_paths.mesh)
        smooth_verts, faces = self.smooth_with_lbo(creation_args, verts, faces)

        self.smooth_mesh_path = os.path.join(self.mesh_dir, self.default_filename + ".off")
        mesh_utils.write_off(self.smooth_mesh_path, smooth_verts, faces)
        
        file_paths.mesh_smooth = self.smooth_mesh_path

        os_utils.write_config_file(self.mesh_dir, self.default_filename, asdict(creation_args)) 

        return file_paths

   
    def smooth_with_lbo(self, lbo_creation_args, verts, faces):
        lbo_data = np.load(lbo_creation_args.lbos_path)
        # the following is to reduce dim in the LBO space (verts * eigenvects * (eigenvects^-1)) :
        projected = np.dot(verts.transpose(), lbo_data["eigenvectors"])
        eigenvects_pinv = np.linalg.pinv(lbo_data["eigenvectors"])
        smooth_verts = np.dot(projected, eigenvects_pinv).transpose() 
        return smooth_verts, faces

class LBOsDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "lbos"
        self.default_filename = "lbo_data" 
    
    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None):
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        self.lbos_dir = os.path.join(self.sample_dir, self.default_dirname)
        os.makedirs(self.lbos_dir, exist_ok=True)

        LBO = LBO_utils.LBOcalc(k=creation_args.num_LBOs, use_torch=creation_args.use_torch, is_point_cloud=creation_args.is_point_cloud)
        self.eigenvectors, self.eigenvalues, self.area_weights = LBO.get_LBOs(*mesh_utils.read_off(creation_args.mesh_path))

        self.lbo_data_path =  os.path.join(self.lbos_dir, f"{creation_args.orig_mesh_name}_{self.default_filename}.npz")
        np.savez(self.lbo_data_path, eigenvectors=self.eigenvectors, eigenvalues=self.eigenvalues, area_weights=self.area_weights)
        
        filename = f"{creation_args.orig_mesh_name}_{self.default_filename}"
        setattr(file_paths, filename, self.lbo_data_path)

        os_utils.write_config_file(self.lbos_dir, filename, asdict(creation_args)) 

        return file_paths

class ConvexMeshDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "meshes"
        self.default_filename = "convex_mesh" 
    

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None):
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        self.mesh_dir = os.path.join(self.sample_dir, self.default_dirname)
        os.makedirs(self.mesh_dir, exist_ok=True)

        verts, faces = mesh_utils.read_off(file_paths.mesh_smooth)
        convex_verts, convex_faces = self.convexify(verts)

        #TODO as 1 func for base class? 
        self.convex_mesh_path = os.path.join(self.mesh_dir, self.default_filename + ".off")
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
        self.default_dirname = "voxels"
        self.default_filename = "voxelized"
        self.prefix = "xyz"

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None):
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        self.voxels_dir = os.path.join(self.sample_dir, self.default_dirname)
        os.makedirs(self.voxels_dir, exist_ok=True)

        mesh_filename = creation_args.mesh_path.split("/")[-1].split(".off")[0]
        voxelized = voxels_mesh_conversions_utils.Mesh2VoxelsConvertor(creation_args.mesh_path, dataset_attrs["shape"]).padded_voxelized

        filename = f"{self.prefix}_{mesh_filename}_{self.default_filename}"
        self.voxelized_mesh_path = os.path.join(self.voxels_dir,  filename + ".npy")
        np.save(self.voxelized_mesh_path, voxelized)
        setattr(file_paths, filename, self.voxelized_mesh_path)

        return file_paths

class TwoDVisDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int, file=None) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "2d_sections_visualization"
        self.default_filename = "2d_sections"
        self.file = file
    

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None):
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        self.output_dir = os.path.join(self.sample_dir, self.default_dirname)
        os.makedirs(self.output_dir, exist_ok=True)

        img_sections_path = os.path.join(self.output_dir, f"scan_{self.default_filename}.jpg") 
        file_paths.scan_sections = img_sections_path
        sections_image = sections_2d_visualization_utils.draw_2d_sections(creation_args.xyz_scan_arr, img_sections_path)
        
        file_paths = sections_2d_visualization_utils.draw_masks_and_contours(sections_image, creation_args.masks_data, self.output_dir, file_paths)
    
        return file_paths

class ThreeDVisDataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int, file=None) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "3d_visualization"
        self.default_smooth_filename = "3d_smooth"
        self.file = file
    

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None):
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        self.output_dir = os.path.join(self.sample_dir, self.default_dirname)
        os.makedirs(self.output_dir, exist_ok=True)

        file_paths.lbo_visualization = mesh_3d_visualization_utils.visualize_grid_of_lbo(   
            verts=creation_args.smooth_mesh_verts, 
            faces=creation_args.smooth_mesh_faces, 
            eigenvectors=creation_args.smooth_mesh_eigenvectors, 
            dirpath=self.output_dir, 
            max_lbos=creation_args.max_smooth_lbo_mesh_visualization,
            mesh_or_pc='mesh',
            prefix=self.default_smooth_filename,)

        file_paths.clear_mesh_visualization = mesh_3d_visualization_utils.visualize_grid_of_lbo(   
            verts=creation_args.smooth_mesh_verts, 
            faces=creation_args.smooth_mesh_faces, 
            eigenvectors=np.ones([creation_args.smooth_mesh_verts.shape[0], 1]), 
            dirpath=self.output_dir, 
            max_lbos=1,
            mesh_or_pc='mesh',
            prefix="smooth_clean_mesh")
        
        
        return file_paths

class H5DataCreator(DataCreator):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "h5_datasets"
        self.default_filename = "dataset" 
    
    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None):
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        self.datasets_dir = os.path.join(self.sample_dir, self.default_dirname)
        os.makedirs(self.datasets_dir, exist_ok=True)

        filename = f"{creation_args.orig_name}_{self.default_filename}"
        self.dataset_path = os.path.join(self.datasets_dir, filename + ".hdf5")
        writing_mode = "w" if creation_args.override else "a"

        mesh_path = getattr(file_paths, creation_args.orig_name)
        vertices, faces = mesh_utils.read_off(mesh_path)
        lbo_data_path = getattr(file_paths, creation_args.orig_name+"_lbo_data")
        lbo_data = np.load(lbo_data_path)

        out_h5 = h5py.File(self.dataset_path, writing_mode)
        
        out_h5.create_dataset(self.name + '_vertices'    , data=vertices                , compression="gzip")
        out_h5.create_dataset(self.name + "_faces"       , data=faces                   , compression="gzip") 
        out_h5.create_dataset(self.name + "_area_weights", data=lbo_data["area_weights"], compression="gzip") 
        out_h5.create_dataset(self.name + "_eigenvectors", data=lbo_data["eigenvectors"], compression="gzip") 
        out_h5.create_dataset(self.name + "_eigenvalus"  , data=lbo_data["eigenvalues"] , compression="gzip")

        out_h5.close()

        setattr(file_paths, filename, self.dataset_path)

        return file_paths





