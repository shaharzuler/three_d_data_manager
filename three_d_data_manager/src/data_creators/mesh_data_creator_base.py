from gettext import npgettext
import os

import numpy as np


from .data_creator_base import DataCreatorBase 
from three_d_data_manager.src.utils import mesh_utils, point_cloud_utils #TODO expose poont cloud utils



class MeshDataCreatorBase(DataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int, creation_args=None) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels, creation_args=creation_args)

    def check_if_exists_default_filename(self) -> bool:
        self.mesh_path = os.path.join(self.subject_dir, self.default_filename + ".off") 
        return os.path.isfile(self.mesh_path)

    def save_mesh_default_filename(self, verts:np.ndarray, faces:np.ndarray) -> None:
        mesh_utils.write_off(self.mesh_path, verts, faces) 

    def save_verts_normals(self, k):
        """treats the verts as point cloud and saves normals to it calculated by pca and k-nn"""
        point_cloud_utils.get_normals(self.verts, k)
