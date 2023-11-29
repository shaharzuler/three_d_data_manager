from dataclasses import asdict
from typing import Dict

import scipy
import numpy as np

from .three_d_arr_data_creator_base import ThreeDArrDataCreatorBase
from three_d_data_manager.src.file_paths import FilePaths
from three_d_data_manager.src.utils import voxels_utils, os_utils


class ZXYVoxelsMaskDataCreator(ThreeDArrDataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int, creation_args=None) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels, creation_args)
        self.default_dirname = "voxels"
        self.default_filename = "zxy_voxels_mask_raw"
    
    def add_sample(self, target_root_dir:str, file_paths:FilePaths, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, dataset_attrs)
        if not self.check_if_exists_default_filename() or self.override:
            output_arr = np.load(self.source_path)
            if voxels_utils.zxy_to_xyz(output_arr).shape != dataset_attrs["shape"]:
                if "voxel_size" in dataset_attrs:
                    voxel_size_zxy = dataset_attrs["voxel_size"][2], dataset_attrs["voxel_size"][0], dataset_attrs["voxel_size"][1]
                    output_arr = scipy.ndimage.zoom(output_arr, voxel_size_zxy)
            output_arr = output_arr.astype(bool)

            self.save_arr_default_filename(output_arr)
        if self.creation_args is not None:
            os_utils.write_config_file(self.subject_dir, self.default_filename, asdict(self.creation_args))

        file_paths.add_path(self.default_filename,  self.sample_name, self.arr_path)

        return file_paths

class ZXYVoxelsExtraMaskDataCreator(ZXYVoxelsMaskDataCreator):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int, creation_args=None) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels, creation_args)
        self.default_filename = "zxy_voxels_extra_mask_raw"
