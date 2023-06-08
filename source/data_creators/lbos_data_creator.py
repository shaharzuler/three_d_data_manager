from dataclasses import asdict
import os

import numpy as np

from .data_creator_base import DataCreatorBase
from three_d_data_manager.source.file_paths import FilePaths
from three_d_data_manager.source.utils import mesh_utils, os_utils, LBO_utils



class LBOsDataCreator(DataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int, creation_args=None) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels, creation_args=creation_args)
        self.default_dirname = "lbos"
        self.default_filename = "lbo_data" 
    
    def add_sample(self, target_root_dir:str, file_paths:FilePaths, dataset_attrs:dict[str,str]=None) -> FilePaths:
        # loading from source_path is not implemented in this class.
        super().add_sample(target_root_dir, dataset_attrs)
        self.lbo_data_path =  os.path.join(self.subject_dir, f"{self.creation_args.orig_mesh_name}_{self.default_filename}.npz")
        filename = f"{self.creation_args.orig_mesh_name}_{self.default_filename}"

        if not self.check_if_exists(self.lbo_data_path) or self.override:
            LBO = LBO_utils.LBOcalc(k=self.creation_args.num_LBOs, use_torch=self.creation_args.use_torch, is_point_cloud=self.creation_args.is_point_cloud)
            self.eigenvectors, self.eigenvalues, self.area_weights = LBO.get_LBOs(*mesh_utils.read_off(self.creation_args.mesh_path))
            np.savez(self.lbo_data_path, eigenvectors=self.eigenvectors, eigenvalues=self.eigenvalues, area_weights=self.area_weights)
            if self.creation_args is not None:
                os_utils.write_config_file(self.subject_dir, filename, asdict(self.creation_args)) 
        file_paths.add_path(filename,  self.sample_name, self.lbo_data_path)

        return file_paths