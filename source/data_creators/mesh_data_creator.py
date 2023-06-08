from dataclasses import asdict

import numpy as np

from .mesh_data_creator_base import MeshDataCreatorBase
from three_d_data_manager.source.file_paths import FilePaths
from three_d_data_manager.source.utils import voxels_mesh_conversions_utils, os_utils, mesh_utils


class MeshDataCreator(MeshDataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int, creation_args=None) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels, creation_args=creation_args)
        self.default_dirname = "meshes"
        self.default_filename = "mesh" 

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, dataset_attrs:dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, dataset_attrs)
        if not self.check_if_exists_default_filename() or self.override:
            if self.source_path is None:
                xyz_voxels_mask_smooth = np.load(file_paths.xyz_voxels_mask_smooth[self.sample_name])
                output_verts, output_faces, normals, values = voxels_mesh_conversions_utils.voxels_mask_to_mesh(xyz_voxels_mask_smooth, self.creation_args.marching_cubes_step_size)
            else:
                output_verts, output_faces = mesh_utils.read_off(self.source_path)
            self.save_mesh_default_filename(output_verts, output_faces)
            if self.creation_args is not None:
                os_utils.write_config_file(self.subject_dir, self.default_filename, asdict(self.creation_args))

        file_paths.add_path("mesh",  self.sample_name, self.mesh_path)

        return file_paths