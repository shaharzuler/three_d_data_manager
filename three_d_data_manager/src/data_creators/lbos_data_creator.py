from dataclasses import asdict
import os
from typing import Dict

import numpy as np
import open3d as o3d

from .data_creator_base import DataCreatorBase
from three_d_data_manager.src.file_paths import FilePaths
from three_d_data_manager.src.utils import mesh_utils, os_utils, LBO_utils



class LBOsDataCreator(DataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int, creation_args=None) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels, creation_args=creation_args)
        self.default_dirname = "lbos"
        self.default_filename = "lbo_data" 
    
    def add_sample(self, target_root_dir:str, file_paths:FilePaths, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        if self.source_path != None:
            raise NotImplementedError("Loading from source path is not supported in this class")
        super().add_sample(target_root_dir, dataset_attrs)
        filename = f"{self.creation_args.orig_geometry_name}_{self.default_filename}"
        self.lbo_data_path =  os.path.join(self.subject_dir, f"{filename}.npz")
        
        if not self.check_if_exists(self.lbo_data_path) or self.override:
            if self.creation_args.is_point_cloud:
                point_cloud =  o3d.io.read_point_cloud(self.creation_args.geometry_path)
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=8, width=0, scale=1.1, linear_fit=True)[0]
                mesh = mesh.simplify_quadric_decimation(2*len(point_cloud.points))
                vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            else:
                vertices, faces = mesh_utils.read_off(self.creation_args.geometry_path)
            LBO = LBO_utils.LBOcalc(k=self.creation_args.num_LBOs, use_torch=self.creation_args.use_torch, is_point_cloud=False)
            self.eigenvectors, self.eigenvalues, self.area_weights = LBO.get_LBOs(vertices, faces)
            np.savez(self.lbo_data_path, eigenvectors=self.eigenvectors, eigenvalues=self.eigenvalues, area_weights=self.area_weights, vertices=vertices, faces=faces)
            if self.creation_args is not None:
                os_utils.write_config_file(self.subject_dir, filename, asdict(self.creation_args)) 
        file_paths.add_path(filename,  self.sample_name, self.lbo_data_path)

        return file_paths