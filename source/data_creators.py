
#TODO split to files (!)

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



class DataCreatorBase:
    def __init__(self, source_path:str, name:str, hirarchy_levels:int, override:bool=True) -> None:
        self.source_path: str = source_path
        self.name: str = name
        self.hirarchy_levels: int = hirarchy_levels
        self.default_top_foldername: str = "orig"
        self.default_dirname = "default_subject_dirname"
        self.override=override

    def add_sample(self, target_root_dir:str, creation_args, dataset_attrs:Dict[str,str]):
        if self.hirarchy_levels>2:
            self.sample_dir = os.path.join(target_root_dir, self.name, *([self.default_top_foldername]*self.hirarchy_levels))
        else: 
            self.sample_dir = os.path.join(target_root_dir, self.name, self.default_top_foldername)
        os.makedirs(self.sample_dir, exist_ok=True)

        self.subject_dir = os.path.join(self.sample_dir, self.default_dirname)
        os.makedirs(self.subject_dir, exist_ok=True)

    def add_sample_from_file(self, file, target_root_dir:str, file_paths:FilePaths, creation_args, dataset_attrs:Dict[str,str]):
        raise NotImplementedError

    def get_properties(self) -> Dict[str, any]:
        return {}

    def check_if_exists(self, filename:str) -> bool:
        return os.path.isfile(filename)


class ThreeDArrDataCreatorBase(DataCreatorBase):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)

    def check_if_exists_default_filename(self) -> bool:
        self.arr_path = os.path.join(self.subject_dir, self.default_filename + ".npy")
        return os.path.isfile(self.arr_path)

    def check_if_exists(self, filename:str) -> bool:
        self.arr_path = filename + ".npy"
        return os.path.isfile(self.arr_path)

    def save_arr_default_filename(self, arr:np.array) -> None:
        np.save(self.arr_path, arr) 

    def save_arr(self, arr:np.array, filename:str) -> None:
        self.arr_path = os.path.join(self.subject_dir, filename + ".npy")
        np.save(self.arr_path, arr)

class MeshDataCreatorBase(DataCreatorBase):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)

    def check_if_exists_default_filename(self) -> bool:
        self.mesh_path = os.path.join(self.subject_dir, self.default_filename + ".off") 
        return os.path.isfile(self.mesh_path)

    def save_mesh_default_filename(self, verts:np.array, faces:np.array) -> None:
        mesh_utils.write_off(self.mesh_path, verts, faces) 


class DicomDataCreator(DataCreatorBase):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "DICOM"
    
    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, creation_args, dataset_attrs)

        dicom_source_file_paths = dicom_utils.get_filepaths_from_img_num(self.source_path, int(self.name))
        self.dicom_target_file_paths = [os.path.join(self.subject_dir, path) for path in dicom_source_file_paths]

        [shutil.copy2(file_path, self.subject_dir) for file_path in dicom_source_file_paths if not self.check_if_exists(os.path.join(self.subject_dir, file_path)) or self.override] 
        
        file_paths.add_path("dicom_dir",  self.name, self.subject_dir)
        file_paths.add_path("dicom_file_paths",  self.name, self.dicom_target_file_paths)
        
        if creation_args is not None:
            os_utils.write_config_file(self.subject_dir, self.default_dirname, asdict(creation_args))

        return file_paths

    def get_properties(self) -> Dict[str, any]:
        slice_path = self.dicom_target_file_paths[0]
        voxel_size = dicom_utils.get_voxel_size(slice_path)
        properties = {
            "voxel_size": voxel_size
        }
        return properties

class XYZArrDataCreator(ThreeDArrDataCreatorBase):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "voxels"
        self.default_filename = "xyz_arr_raw"
    
    def get_xyz_arr_from_dicom(self, dicom_dir:str):
        self.xyz_arr = dicom_utils.images_to_3d_arr(dicom_dir, int(self.name))
        return self.xyz_arr

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        if not self.check_if_exists_default_filename() or self.override:
            output_arr = self.get_xyz_arr_from_dicom(file_paths.dicom_dir[self.name])
            self.save_arr_default_filename(output_arr)
            if creation_args is not None:
                os_utils.write_config_file(self.subject_dir, self.default_filename, asdict(creation_args))
        else: 
            self.xyz_arr = np.load(self.arr_path)

        file_paths.add_path("xyz_arr",  self.name, self.arr_path)

        return file_paths

    def get_properties(self) -> dict:
        properties = {
            "shape": self.xyz_arr.shape
        }
        return properties

class XYZVoxelsMaskDataCreator(ThreeDArrDataCreatorBase):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int, file=None) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "voxels"
        self.default_filename = "xyz_voxels_mask_raw"

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        if not self.check_if_exists_default_filename() or self.override:
            zxy_voxels_mask_arr = np.load(file_paths.zxy_voxels_mask_raw[self.name])
            output_arr = voxels_utils.zxy_to_xyz(zxy_voxels_mask_arr)

            self.save_arr_default_filename(output_arr)

            if creation_args is not None:
                os_utils.write_config_file(self.subject_dir, self.default_filename, asdict(creation_args))

        file_paths.add_path("xyz_voxels_mask_raw",  self.name, self.arr_path)

        return file_paths

class ZXYVoxelsMaskDataCreator(ThreeDArrDataCreatorBase):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "voxels"
        self.default_filename = "zxy_voxels_mask_raw"
    
    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        if not self.check_if_exists_default_filename() or self.override:
            output_arr = np.load(self.source_path)
            if voxels_utils.zxy_to_xyz(output_arr).shape != dataset_attrs["shape"]:
                if "voxel_size" in dataset_attrs:
                    voxel_size_zxy = dataset_attrs["voxel_size"][2], dataset_attrs["voxel_size"][0], dataset_attrs["voxel_size"][1]
                    output_arr = scipy.ndimage.zoom(output_arr, voxel_size_zxy)
            output_arr = output_arr.astype(bool)

            self.save_arr_default_filename(output_arr)
        if creation_args is not None:
            os_utils.write_config_file(self.subject_dir, self.default_filename, asdict(creation_args))

        file_paths.add_path("zxy_voxels_mask_raw",  self.name, self.arr_path)

        return file_paths

class SmoothVoxelsMaskDataCreator(ThreeDArrDataCreatorBase):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int, file=None) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "voxels"
        self.default_filename = "xyz_voxels_mask_smooth"

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        if not self.check_if_exists_default_filename() or self.override:
            xyz_voxels_mask_arr = np.load(file_paths.xyz_voxels_mask_raw[self.name])
            xyz_voxels_mask_smooth = voxels_utils.fill_holes(masks_arr=xyz_voxels_mask_arr, area_threshold=creation_args.fill_holes_Area_threshold)
            xyz_voxels_mask_smooth = voxels_utils.voxel_smoothing(xyz_voxels_mask_smooth, creation_args.opening_footprint_radius, creation_args.closing_to_opening_ratio) 
            output_arr = voxels_utils.fill_holes(masks_arr=xyz_voxels_mask_smooth, area_threshold=creation_args.fill_holes_Area_threshold)

            self.save_arr_default_filename(output_arr)
            if creation_args is not None:
                os_utils.write_config_file(self.subject_dir, self.default_filename, asdict(creation_args)) 

        file_paths.add_path("xyz_voxels_mask_smooth",  self.name, self.arr_path)

        return file_paths

class MeshDataCreator(MeshDataCreatorBase):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int, file=None) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "meshes"
        self.default_filename = "mesh" 

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        if not self.check_if_exists_default_filename() or self.override:
            xyz_voxels_mask_smooth = np.load(file_paths.xyz_voxels_mask_smooth[self.name])
            output_verts, output_faces, normals, values = voxels_mesh_conversions_utils.voxels_mask_to_mesh(xyz_voxels_mask_smooth, creation_args.marching_cubes_step_size)

            self.save_mesh_default_filename(output_verts, output_faces)
            if creation_args is not None:
                os_utils.write_config_file(self.subject_dir, self.default_filename, asdict(creation_args))

        file_paths.add_path("mesh",  self.name, self.mesh_path)

        return file_paths

class SmoothLBOMeshDataCreator(MeshDataCreatorBase):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "meshes"
        self.default_filename = "smooth_mesh"     

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        if not self.check_if_exists_default_filename() or self.override:
            verts, faces = mesh_utils.read_off(file_paths.mesh[self.name])
            output_verts, output_faces = self._smooth_with_lbo(creation_args, verts, faces)

            self.save_mesh_default_filename(output_verts, output_faces)
            if creation_args is not None:
                os_utils.write_config_file(self.subject_dir, self.default_filename, asdict(creation_args)) 
        file_paths.add_path("mesh_smooth",  self.name, self.mesh_path)

        return file_paths

   
    def _smooth_with_lbo(self, lbo_creation_args, verts:np.array, faces:np.array):
        lbo_data = np.load(lbo_creation_args.lbos_path)
        # the following is to reduce dim in the LBO space (verts * eigenvects * (eigenvects^-1)) :
        projected = np.dot(verts.transpose(), lbo_data["eigenvectors"])
        eigenvects_pinv = np.linalg.pinv(lbo_data["eigenvectors"])
        smooth_verts = np.dot(projected, eigenvects_pinv).transpose() 
        return smooth_verts, faces

class ConvexMeshDataCreator(MeshDataCreatorBase):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "meshes"
        self.default_filename = "convex_mesh" 
    

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        if not self.check_if_exists_default_filename() or self.override:
            verts, faces = mesh_utils.read_off(file_paths.mesh_smooth[self.name])
            output_verts, output_faces = self._convexify(verts)

            self.save_mesh_default_filename(output_verts, output_faces)
            if creation_args is not None:
                os_utils.write_config_file(self.subject_dir, self.default_filename, asdict(creation_args))
        file_paths.add_path("mesh_convex",  self.name, self.mesh_path)

        return file_paths

    def _convexify(self, vertices):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        convex_hull, orig_pcd_point_indices = pcd.compute_convex_hull()
        verts_convex_hull = np.asarray(convex_hull.vertices).astype(np.float32)
        faces_convex_hull = np.asarray(convex_hull.triangles).astype(np.int64)
        return verts_convex_hull, faces_convex_hull

class LBOsDataCreator(DataCreatorBase):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "lbos"
        self.default_filename = "lbo_data" 
    
    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        self.lbo_data_path =  os.path.join(self.subject_dir, f"{creation_args.orig_mesh_name}_{self.default_filename}.npz")
        filename = f"{creation_args.orig_mesh_name}_{self.default_filename}"

        if not self.check_if_exists(self.lbo_data_path) or self.override:
            LBO = LBO_utils.LBOcalc(k=creation_args.num_LBOs, use_torch=creation_args.use_torch, is_point_cloud=creation_args.is_point_cloud)
            self.eigenvectors, self.eigenvalues, self.area_weights = LBO.get_LBOs(*mesh_utils.read_off(creation_args.mesh_path))
            np.savez(self.lbo_data_path, eigenvectors=self.eigenvectors, eigenvalues=self.eigenvalues, area_weights=self.area_weights)
            if creation_args is not None:
                os_utils.write_config_file(self.subject_dir, filename, asdict(creation_args)) 
        file_paths.add_path(filename,  self.name, self.lbo_data_path)

        return file_paths

class VoxelizedMeshDataCreator(ThreeDArrDataCreatorBase):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "voxels"
        self.default_filename = "voxelized"
        self.prefix = "xyz"

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        mesh_filename = creation_args.mesh_path.split("/")[-1].split(".off")[0]
        filename = f"{self.prefix}_{mesh_filename}_{self.default_filename}"
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        if not self.check_if_exists(filename) or self.override:
            output_arr = voxels_mesh_conversions_utils.Mesh2VoxelsConvertor(creation_args.mesh_path, dataset_attrs["shape"]).padded_voxelized
            self.save_arr(output_arr, filename) 
            if creation_args is not None:
                os_utils.write_config_file(self.subject_dir, f"{self.prefix}_{self.default_filename}", asdict(creation_args))

        file_paths.add_path(filename,  self.name, self.arr_path)

        return file_paths

class TwoDVisDataCreator(DataCreatorBase):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int, file=None) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "2d_sections_visualization"
        self.default_filename = "2d_sections"

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None) -> FilePaths: #override will not work here
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        img_sections_path = os.path.join(self.subject_dir, f"scan_{self.default_filename}.jpg") 
        file_paths.add_path("scan_sections", self.name, img_sections_path)
        sections_image = sections_2d_visualization_utils.draw_2d_sections(creation_args.xyz_scan_arr, img_sections_path)
        file_paths = sections_2d_visualization_utils.draw_masks_and_contours(sections_image, creation_args.masks_data, self.subject_dir, file_paths, self.name)

        if creation_args is not None:
            os_utils.write_config_file(self.subject_dir, self.default_filename, asdict(creation_args))
    
        return file_paths

class ThreeDVisDataCreator(DataCreatorBase):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int, file=None) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "3d_visualization"
        self.default_smooth_filename = "3d_smooth"

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None) -> FilePaths: # will override anyway
        super().add_sample(target_root_dir, creation_args, dataset_attrs) 
        
        lbo_visualization_path = mesh_3d_visualization_utils.visualize_grid_of_lbo(   
            verts=creation_args.smooth_mesh_verts, 
            faces=creation_args.smooth_mesh_faces, 
            eigenvectors=creation_args.smooth_mesh_eigenvectors, 
            dirpath=self.subject_dir, 
            max_lbos=creation_args.max_smooth_lbo_mesh_visualization,
            mesh_or_pc='mesh',
            prefix=self.default_smooth_filename)
        file_paths.add_path("lbo_visualization", self.name, lbo_visualization_path)

        clear_mesh_visualization_path = mesh_3d_visualization_utils.visualize_grid_of_lbo(   
            verts=creation_args.smooth_mesh_verts, 
            faces=creation_args.smooth_mesh_faces, 
            eigenvectors=np.ones([creation_args.smooth_mesh_verts.shape[0], 1]), 
            dirpath=self.subject_dir, 
            max_lbos=1,
            mesh_or_pc='mesh',
            prefix="smooth_clean_mesh")
        file_paths.add_path("clear_mesh_visualization", self.name, clear_mesh_visualization_path)
        
        if creation_args is not None:
            os_utils.write_config_file(self.subject_dir, self.default_smooth_filename, asdict(creation_args))

        return file_paths

class H5DataCreator(DataCreatorBase):
    def __init__(self, source_path:str, name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, name, hirarchy_levels)
        self.default_dirname = "h5_datasets"
        self.default_filename = "dataset" 
    
    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, creation_args, dataset_attrs)

        filename = f"{creation_args.orig_name}_{self.default_filename}"
        self.dataset_path = os.path.join(self.subject_dir, filename + ".hdf5")
        writing_mode = "w" if creation_args.override else "a" # override works differently here. it will perform and append if file exists. existing keys will be overwritten anyway.

        mesh_path = getattr(file_paths, creation_args.orig_name)[self.name]
        vertices, faces = mesh_utils.read_off(mesh_path)
        lbo_data_path = getattr(file_paths, creation_args.orig_name+"_lbo_data")[self.name]
        lbo_data = np.load(lbo_data_path)

        out_h5 = h5py.File(self.dataset_path, writing_mode)
        
        out_h5.create_dataset(self.name + '_vertices'    , data=vertices                , compression="gzip")
        out_h5.create_dataset(self.name + "_faces"       , data=faces                   , compression="gzip") 
        out_h5.create_dataset(self.name + "_area_weights", data=lbo_data["area_weights"], compression="gzip") 
        out_h5.create_dataset(self.name + "_eigenvectors", data=lbo_data["eigenvectors"], compression="gzip") 
        out_h5.create_dataset(self.name + "_eigenvalus"  , data=lbo_data["eigenvalues"] , compression="gzip")

        out_h5.close()

        if creation_args is not None:
            os_utils.write_config_file(self.subject_dir, filename, asdict(creation_args))

        file_paths.add_path(filename, self.name, self.dataset_path)

        return file_paths





