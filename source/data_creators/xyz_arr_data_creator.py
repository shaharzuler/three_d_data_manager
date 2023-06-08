
from dataclasses import asdict

import numpy as np

from .three_d_arr_data_creator_base import ThreeDArrDataCreatorBase
from three_d_data_manager.source.file_paths import FilePaths
from three_d_data_manager.source.utils import dicom_utils, os_utils



class XYZArrDataCreator(ThreeDArrDataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels)
        self.default_dirname = "voxels"
        self.default_filename = "xyz_arr_raw"
    
    def get_xyz_arr_from_dicom(self, dicom_dir:str):
        self.xyz_arr = dicom_utils.images_to_3d_arr(dicom_dir, int(self.sample_name))
        return self.xyz_arr

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, creation_args, dataset_attrs)
        if not self.check_if_exists_default_filename() or self.override:
            if self.source_path is None:
                output_arr = self.get_xyz_arr_from_dicom(file_paths.dicom_dir[self.sample_name])
            else:
                output_arr = np.load(self.source_path) #doesnt take care of scaling and orientation.
            self.save_arr_default_filename(output_arr)
            if creation_args is not None:
                os_utils.write_config_file(self.subject_dir, self.default_filename, asdict(creation_args))
        else: 
            self.xyz_arr = np.load(self.arr_path)

        file_paths.add_path("xyz_arr",  self.sample_name, self.arr_path)

        return file_paths

    def get_properties(self) -> dict:
        properties = {
            "shape": self.xyz_arr.shape
        }
        return properties