
from dataclasses import asdict
import os

from .data_creator_base import DataCreatorBase
from three_d_data_manager.source.file_paths import FilePaths
from three_d_data_manager.source.utils import sections_2d_visualization_utils, os_utils



class TwoDVisDataCreator(DataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int, file=None) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels)
        self.default_dirname = "2d_sections_visualization"
        self.default_filename = "2d_sections"

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:dict[str,str]=None) -> FilePaths: #override will not work here
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        img_sections_path = os.path.join(self.subject_dir, f"scan_{self.default_filename}.jpg") 
        file_paths.add_path("scan_sections", self.sample_name, img_sections_path)
        sections_image = sections_2d_visualization_utils.draw_2d_sections(creation_args.xyz_scan_arr, img_sections_path)
        file_paths = sections_2d_visualization_utils.draw_masks_and_contours(sections_image, creation_args.masks_data, self.subject_dir, file_paths, self.sample_name)

        if creation_args is not None:
            os_utils.write_config_file(self.subject_dir, self.default_filename, asdict(creation_args))
    
        return file_paths
