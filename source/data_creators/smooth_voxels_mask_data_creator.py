
from dataclasses import asdict

import numpy as np

from three_d_data_manager.source.data_creators.three_d_arr_data_creator_base import ThreeDArrDataCreatorBase
from three_d_data_manager.source.file_paths import FilePaths
from three_d_data_manager.source.utils import voxels_utils, os_utils


class SmoothVoxelsMaskDataCreator(ThreeDArrDataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int, file=None) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels)
        self.default_dirname = "voxels"
        self.default_filename = "xyz_voxels_mask_smooth"

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        if not self.check_if_exists_default_filename() or self.override:
            if self.source_path is None:
                xyz_voxels_mask_arr = np.load(file_paths.xyz_voxels_mask_raw[self.sample_name])
                xyz_voxels_mask_smooth = voxels_utils.fill_holes(masks_arr=xyz_voxels_mask_arr, area_threshold=creation_args.fill_holes_Area_threshold)
                xyz_voxels_mask_smooth = voxels_utils.voxel_smoothing(xyz_voxels_mask_smooth, creation_args.opening_footprint_radius, creation_args.closing_to_opening_ratio) 
                output_arr = voxels_utils.fill_holes(masks_arr=xyz_voxels_mask_smooth, area_threshold=creation_args.fill_holes_Area_threshold)
            else:
                output_arr = np.load(self.source_path)
            self.save_arr_default_filename(output_arr)
            if creation_args is not None:
                os_utils.write_config_file(self.subject_dir, self.default_filename, asdict(creation_args)) 

        file_paths.add_path("xyz_voxels_mask_smooth",  self.sample_name, self.arr_path)

        return file_paths