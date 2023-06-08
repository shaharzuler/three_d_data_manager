from dataclasses import asdict

import open3d as o3d
import numpy as np

from .mesh_data_creator_base import MeshDataCreatorBase
from three_d_data_manager.source.file_paths import FilePaths
from three_d_data_manager.source.utils import mesh_utils, os_utils


class ConvexMeshDataCreator(MeshDataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels)
        self.default_dirname = "meshes"
        self.default_filename = "convex_mesh" 
    

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        if not self.check_if_exists_default_filename() or self.override:
            if self.source_path is None:
                verts, faces = mesh_utils.read_off(file_paths.mesh_smooth[self.sample_name])
                output_verts, output_faces = self._convexify(verts)
            else:
                output_verts, output_faces = mesh_utils.read_off(self.source_path)

            self.save_mesh_default_filename(output_verts, output_faces)
            if creation_args is not None:
                os_utils.write_config_file(self.subject_dir, self.default_filename, asdict(creation_args))
        file_paths.add_path("mesh_convex",  self.sample_name, self.mesh_path)

        return file_paths

    def _convexify(self, vertices):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        convex_hull, orig_pcd_point_indices = pcd.compute_convex_hull()
        verts_convex_hull = np.asarray(convex_hull.vertices).astype(np.float32)
        faces_convex_hull = np.asarray(convex_hull.triangles).astype(np.int64)
        return verts_convex_hull, faces_convex_hull
