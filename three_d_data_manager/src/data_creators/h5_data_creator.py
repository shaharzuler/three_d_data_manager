from dataclasses import asdict
import os

import numpy as np
import h5py
import open3d as o3d


from .data_creator_base import DataCreatorBase
from three_d_data_manager.src.file_paths import FilePaths
from three_d_data_manager.src.utils import mesh_utils, os_utils


class H5DataCreator(DataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int, creation_args=None) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels, creation_args=creation_args)
        self.default_dirname = "h5_datasets"
        self.default_filename = "dataset" 
    
    def add_sample(self, target_root_dir:str, file_paths:FilePaths, dataset_attrs:dict[str,str]=None) -> FilePaths:
        if self.source_path != None:
            raise NotImplementedError("Loading from source path is not supported in this class")
        super().add_sample(target_root_dir, dataset_attrs)

        filename = f"{self.creation_args.orig_name}_{self.default_filename}"
        self.dataset_path = os.path.join(self.subject_dir, filename + ".hdf5")
        writing_mode = "w" if self.creation_args.override else "a" # Override works differently here. It will perform and append if file exists. Existing keys will be overwritten anyway.

        geometry_path = getattr(file_paths, self.creation_args.orig_name)[self.sample_name]
        lbo_data_path = getattr(file_paths, self.creation_args.orig_name+"_lbo_data")[self.sample_name]
        lbo_data = np.load(lbo_data_path)
        if self.creation_args.is_point_cloud:
            points = np.asarray(o3d.io.read_point_cloud(geometry_path).points)
            vertices, faces = lbo_data["vertices"], lbo_data["faces"]
        else:
            points = np.array([])
            vertices, faces = mesh_utils.read_off(geometry_path)
        
        out_h5 = h5py.File(self.dataset_path, writing_mode)
        
        out_h5.create_dataset(self.sample_name + '_points'      , data=points                  , compression="gzip")
        out_h5.create_dataset(self.sample_name + '_vertices'    , data=vertices                , compression="gzip")
        out_h5.create_dataset(self.sample_name + "_faces"       , data=faces                   , compression="gzip") 
        out_h5.create_dataset(self.sample_name + "_area_weights", data=lbo_data["area_weights"], compression="gzip") 
        out_h5.create_dataset(self.sample_name + "_eigenvectors", data=lbo_data["eigenvectors"], compression="gzip") 
        out_h5.create_dataset(self.sample_name + "_eigenvalues"  , data=lbo_data["eigenvalues"] , compression="gzip")

        out_h5.close()

        if self.creation_args is not None:
            os_utils.write_config_file(self.subject_dir, filename, asdict(self.creation_args))

        file_paths.add_path(filename, self.sample_name, self.dataset_path)

        return file_paths