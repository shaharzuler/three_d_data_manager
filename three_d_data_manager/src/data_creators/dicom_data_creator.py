from dataclasses import asdict
import shutil
import os

from .data_creator_base import DataCreatorBase
from three_d_data_manager.src.file_paths import FilePaths
from three_d_data_manager.src.utils import dicom_utils, os_utils



class DicomDataCreator(DataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int, creation_args=None) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels, creation_args=creation_args)
        self.default_dirname = "DICOM"
    
    def add_sample(self, target_root_dir:str, file_paths:FilePaths, dataset_attrs:dict[str,str]=None) -> FilePaths:
        super().add_sample(target_root_dir, dataset_attrs)

        try:
            dicom_source_file_paths = dicom_utils.get_filepaths_from_img_num(self.source_path, int(self.sample_name))
        except ValueError:
            dicom_source_file_paths = self.source_path
        self.dicom_target_file_paths = [os.path.join(self.subject_dir, path) for path in dicom_source_file_paths]

        [shutil.copy2(file_path, self.subject_dir) for file_path in dicom_source_file_paths if not self.check_if_exists(os.path.join(self.subject_dir, file_path)) or self.override] 
        
        file_paths.add_path("dicom_dir",  self.sample_name, self.subject_dir)
        file_paths.add_path("dicom_file_paths",  self.sample_name, self.dicom_target_file_paths)
        
        if self.creation_args is not None:
            os_utils.write_config_file(self.subject_dir, self.default_dirname, asdict(self.creation_args))

        return file_paths

    def get_properties(self) -> dict[str, any]:
        slice_path = self.dicom_target_file_paths[0]
        voxel_size = dicom_utils.get_voxel_size(slice_path)
        properties = {
            "voxel_size": voxel_size
        }
        return properties