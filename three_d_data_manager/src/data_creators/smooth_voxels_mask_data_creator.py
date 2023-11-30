
from dataclasses import asdict
from typing import Dict

import numpy as np

from three_d_data_manager.src.data_creators.three_d_arr_data_creator_base import ThreeDArrDataCreatorBase
from three_d_data_manager.src.file_paths import FilePaths
from three_d_data_manager.src.utils import voxels_utils, os_utils


class SmoothVoxelsMaskDataCreator(ThreeDArrDataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int, creation_args=None) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels, creation_args)
        self.default_dirname = "voxels"
        self.default_filename = "xyz_voxels_mask_smooth"

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, dataset_attrs:Dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, dataset_attrs)
        if not self.check_if_exists_default_filename() or self.override:
            if self.source_path is None:
                source_path = self._get_source_path(file_paths)
                xyz_voxels_mask_arr = np.load(source_path)
                xyz_voxels_mask_smooth = voxels_utils.fill_holes(masks_arr=xyz_voxels_mask_arr, area_threshold=self.creation_args.fill_holes_Area_threshold)
                xyz_voxels_mask_smooth = voxels_utils.voxel_smoothing(xyz_voxels_mask_smooth, self.creation_args.opening_footprint_radius, self.creation_args.closing_to_opening_ratio) 
                output_arr = voxels_utils.fill_holes(masks_arr=xyz_voxels_mask_smooth, area_threshold=self.creation_args.fill_holes_Area_threshold)
            else:
                output_arr = np.load(self.source_path)
            self.save_arr_default_filename(output_arr)
            if self.creation_args is not None:
                os_utils.write_config_file(self.subject_dir, self.default_filename, asdict(self.creation_args)) 

        file_paths.add_path(self.default_filename,  self.sample_name, self.arr_path)

        return file_paths

    def _get_source_path(self, file_paths):
        return file_paths.xyz_voxels_mask_raw[self.sample_name]

class SmoothVoxelsExtraMaskDataCreator(SmoothVoxelsMaskDataCreator):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int, creation_args=None) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels, creation_args)
        self.default_filename = "xyz_voxels_extra_mask_smooth"

    def _get_source_path(self, file_paths):
        return file_paths.xyz_voxels_extra_mask_smooth[self.sample_name]    