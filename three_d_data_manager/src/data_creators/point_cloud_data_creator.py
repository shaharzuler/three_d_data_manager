from dataclasses import asdict
import os
from typing import Dict

import open3d as o3d

from .mesh_data_creator_base import DataCreatorBase
from three_d_data_manager.src.file_paths import FilePaths
from three_d_data_manager.src.utils import voxels_mesh_conversions_utils, os_utils, mesh_utils


class PointCloudDataCreator(DataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int, creation_args=None) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels, creation_args=creation_args)
        self.default_dirname = "point_clouds"
        self.default_filename = "point_cloud" 

    def check_if_exists_default_filename(self) -> bool:
        self.filename = f"{self.default_filename}_from_{self.creation_args.orig_mesh_name}"
        self.point_cloud_path = os.path.join(self.subject_dir, f"{self.filename}.ply") 
        return os.path.isfile(self.point_cloud_path)

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, dataset_attrs)
        if not self.check_if_exists_default_filename() or self.override:
            if self.source_path is None:
                mesh_o3d = o3d.io.read_triangle_mesh(self.creation_args.mesh_path)#file_paths.mesh[self.sample_name])
                mesh_o3d.compute_vertex_normals()
                point_cloud = mesh_o3d.sample_points_uniformly(number_of_points=int(self.creation_args.num_points))
            else:
                point_cloud =  o3d.io.read_point_cloud(self.point_cloud_path)
            o3d.io.write_point_cloud(self.point_cloud_path, point_cloud)
            if self.creation_args is not None:
                os_utils.write_config_file(self.subject_dir, self.filename, asdict(self.creation_args))

        file_paths.add_path(self.filename,  self.sample_name, self.point_cloud_path)

        return file_paths