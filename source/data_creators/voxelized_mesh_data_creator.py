from dataclasses import asdict

import numpy as np

from .three_d_arr_data_creator_base import ThreeDArrDataCreatorBase
from three_d_data_manager.source.file_paths import FilePaths
from three_d_data_manager.source.utils import voxels_mesh_conversions_utils, os_utils


class VoxelizedMeshDataCreator(ThreeDArrDataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels)
        self.default_dirname = "voxels"
        self.default_filename = "voxelized"
        self.prefix = "xyz"

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:dict[str,str]=None) -> FilePaths:
        mesh_filename = creation_args.mesh_path.split("/")[-1].split(".off")[0]
        filename = f"{self.prefix}_{mesh_filename}_{self.default_filename}"
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        if not self.check_if_exists(filename) or self.override:
            if self.source_path is None:
                output_arr = voxels_mesh_conversions_utils.Mesh2VoxelsConvertor(creation_args.mesh_path, dataset_attrs["shape"]).padded_voxelized
            else:
                output_arr = np.load(self.source_path)
            self.save_arr(output_arr, filename) 
            if creation_args is not None:
                os_utils.write_config_file(self.subject_dir, f"{self.prefix}_{self.default_filename}", asdict(creation_args))

        file_paths.add_path(filename,  self.sample_name, self.arr_path)

        return file_paths