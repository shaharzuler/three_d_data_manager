from dataclasses import asdict
import numpy as np

from .data_creator_base import DataCreatorBase
from three_d_data_manager.source.file_paths import FilePaths
from three_d_data_manager.source.utils import mesh_3d_visualization_utils, os_utils


class ThreeDVisDataCreator(DataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int, file=None) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels)
        self.default_dirname = "3d_visualization"
        self.default_smooth_filename = "3d_smooth"

    def add_sample(self, target_root_dir:str, file_paths:FilePaths, creation_args=None, dataset_attrs:dict[str,str]=None) -> FilePaths: # will override anyway
        super().add_sample(target_root_dir, creation_args, dataset_attrs) 
        
        lbo_visualization_path = mesh_3d_visualization_utils.visualize_grid_of_lbo(   
            verts=creation_args.smooth_mesh_verts, 
            faces=creation_args.smooth_mesh_faces, 
            eigenvectors=creation_args.smooth_mesh_eigenvectors, 
            dirpath=self.subject_dir, 
            max_lbos=creation_args.max_smooth_lbo_mesh_visualization,
            mesh_or_pc='mesh',
            prefix=self.default_smooth_filename)
        file_paths.add_path("lbo_visualization", self.sample_name, lbo_visualization_path)

        clear_mesh_visualization_path = mesh_3d_visualization_utils.visualize_grid_of_lbo(   
            verts=creation_args.smooth_mesh_verts, 
            faces=creation_args.smooth_mesh_faces, 
            eigenvectors=np.ones([creation_args.smooth_mesh_verts.shape[0], 1]), 
            dirpath=self.subject_dir, 
            max_lbos=1,
            mesh_or_pc='mesh',
            prefix="smooth_clean_mesh")
        file_paths.add_path("clear_mesh_visualization", self.sample_name, clear_mesh_visualization_path)
        
        if creation_args is not None:
            os_utils.write_config_file(self.subject_dir, self.default_smooth_filename, asdict(creation_args))

        return file_paths