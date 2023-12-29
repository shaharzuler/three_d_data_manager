from dataclasses import asdict
import os
from typing import Dict

import open3d as o3d
import numpy as np

from .mesh_data_creator_base import DataCreatorBase
from three_d_data_manager.src.file_paths import FilePaths
from three_d_data_manager.src.utils import os_utils


class VertexNormalsDataCreator(DataCreatorBase): # TODO add to readme and example usage. todo add visualization?
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int, creation_args=None) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels, creation_args=creation_args)
        self.default_dirname = "vertices_normals"
        self.default_filename = "vertices_normals" 

    def check_if_exists_default_filename(self) -> bool:
        self.filename = f"{self.default_filename}_from_{self.creation_args.orig_geometry_name}"
        self.normals_path = os.path.join(self.subject_dir, f"{self.filename}.ply") 
        return os.path.isfile(self.normals_path)

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, dataset_attrs)
        if not self.check_if_exists_default_filename() or self.override:
            if self.source_path is None:
                geometry = o3d.io.read_triangle_mesh(self.creation_args.geometry_path) # TODO add option for point cloud maybe try catch
                geometry.compute_vertex_normals(normalized=True)
                normals = np.asarray(geometry.vertex_normals)
                point_cloud_scruct = o3d.geometry.PointCloud()
                point_cloud_scruct.points = o3d.utility.Vector3dVector(normals)
            else:
                normals =  o3d.io.read_point_cloud(self.normals_path)
            o3d.io.write_point_cloud(self.normals_path, point_cloud_scruct) # taking advantage of pointcloud data structure for saving this normals file.
            if self.creation_args is not None:
                os_utils.write_config_file(self.subject_dir, self.filename, asdict(self.creation_args))

        file_paths.add_path(self.filename,  self.sample_name, self.normals_path)

        return file_paths