from dataclasses import asdict
import numpy as np
import h5py
import os

from .data_creator_base import DataCreatorBase
from three_d_data_manager.source.file_paths import FilePaths
from three_d_data_manager.source.utils import mesh_utils, os_utils


class H5DataCreator(DataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels)
        self.default_dirname = "h5_datasets"
        self.default_filename = "dataset" 
    
    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:dict[str,str]=None) -> FilePaths:
        # loading from source_path is not implemented for this class
        super().add_sample(target_root_dir, creation_args, dataset_attrs)

        filename = f"{creation_args.orig_name}_{self.default_filename}"
        self.dataset_path = os.path.join(self.subject_dir, filename + ".hdf5")
        writing_mode = "w" if creation_args.override else "a" # override works differently here. it will perform and append if file exists. existing keys will be overwritten anyway.

        mesh_path = getattr(file_paths, creation_args.orig_name)[self.sample_name]
        vertices, faces = mesh_utils.read_off(mesh_path)
        lbo_data_path = getattr(file_paths, creation_args.orig_name+"_lbo_data")[self.sample_name]
        lbo_data = np.load(lbo_data_path)

        out_h5 = h5py.File(self.dataset_path, writing_mode)
        
        out_h5.create_dataset(self.sample_name + '_vertices'    , data=vertices                , compression="gzip")
        out_h5.create_dataset(self.sample_name + "_faces"       , data=faces                   , compression="gzip") 
        out_h5.create_dataset(self.sample_name + "_area_weights", data=lbo_data["area_weights"], compression="gzip") 
        out_h5.create_dataset(self.sample_name + "_eigenvectors", data=lbo_data["eigenvectors"], compression="gzip") 
        out_h5.create_dataset(self.sample_name + "_eigenvalus"  , data=lbo_data["eigenvalues"] , compression="gzip")

        out_h5.close()

        if creation_args is not None:
            os_utils.write_config_file(self.subject_dir, filename, asdict(creation_args))

        file_paths.add_path(filename, self.sample_name, self.dataset_path)

        return file_paths