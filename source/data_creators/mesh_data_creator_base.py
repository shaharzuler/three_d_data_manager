from gettext import npgettext
import os

import numpy as np

from .data_creator_base import DataCreatorBase 
from three_d_data_manager.source.utils import mesh_utils



class MeshDataCreatorBase(DataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels)

    def check_if_exists_default_filename(self) -> bool:
        self.mesh_path = os.path.join(self.subject_dir, self.default_filename + ".off") 
        return os.path.isfile(self.mesh_path)

    def save_mesh_default_filename(self, verts:np.array, faces:np.array) -> None:
        mesh_utils.write_off(self.mesh_path, verts, faces) 