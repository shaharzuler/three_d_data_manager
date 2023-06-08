from dataclasses import asdict

import numpy as np

from .mesh_data_creator_base import MeshDataCreatorBase
from three_d_data_manager.source.file_paths import FilePaths
from three_d_data_manager.source.utils import mesh_utils, os_utils



class SmoothLBOMeshDataCreator(MeshDataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels)
        self.default_dirname = "meshes"
        self.default_filename = "smooth_mesh"     

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        if not self.check_if_exists_default_filename() or self.override:
            if self.source_path is None:
                verts, faces = mesh_utils.read_off(file_paths.mesh[self.sample_name])
                output_verts, output_faces = self._smooth_with_lbo(creation_args, verts, faces)
            else:
                output_verts, output_faces = mesh_utils.read_off(self.source_path)

            self.save_mesh_default_filename(output_verts, output_faces)
            if creation_args is not None:
                os_utils.write_config_file(self.subject_dir, self.default_filename, asdict(creation_args)) 
        file_paths.add_path("mesh_smooth",  self.sample_name, self.mesh_path)

        return file_paths

   
    def _smooth_with_lbo(self, lbo_creation_args, verts:np.array, faces:np.array):
        lbo_data = np.load(lbo_creation_args.lbos_path)
        # the following is to reduce dim in the LBO space (verts * eigenvects * (eigenvects^-1)) :
        projected = np.dot(verts.transpose(), lbo_data["eigenvectors"])
        eigenvects_pinv = np.linalg.pinv(lbo_data["eigenvectors"])
        smooth_verts = np.dot(projected, eigenvects_pinv).transpose() 
        return smooth_verts, faces
