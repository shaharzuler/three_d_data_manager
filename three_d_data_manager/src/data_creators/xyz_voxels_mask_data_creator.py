from dataclasses import asdict

import numpy as np

from .three_d_arr_data_creator_base import ThreeDArrDataCreatorBase
from three_d_data_manager.src.file_paths import FilePaths
from three_d_data_manager.src.utils import voxels_utils, os_utils


class XYZVoxelsMaskDataCreator(ThreeDArrDataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int, creation_args=None) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels, creation_args)
        self.default_dirname = "voxels"
        self.default_filename = "xyz_voxels_mask_raw"

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, dataset_attrs:dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, dataset_attrs)
        if not self.check_if_exists_default_filename() or self.override:
            if self.source_path is None:
                zxy_voxels_mask_arr = np.load(file_paths.zxy_voxels_mask_raw[self.sample_name])
                output_arr = voxels_utils.zxy_to_xyz(zxy_voxels_mask_arr)
            else:
                output_arr = np.load(self.source_path)

            self.save_arr_default_filename(output_arr)

            if self.creation_args is not None:
                os_utils.write_config_file(self.subject_dir, self.default_filename, asdict(self.creation_args))

        file_paths.add_path("xyz_voxels_mask_raw",  self.sample_name, self.arr_path)

        return file_paths