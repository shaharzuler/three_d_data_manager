import os

import numpy as np

from .data_creator_base import DataCreatorBase
from three_d_data_manager.src.file_paths import FilePaths


class ThreeDArrDataCreatorBase(DataCreatorBase):
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int, creation_args=None) -> None:
        super().__init__(source_path, sample_name, hirarchy_levels, creation_args=creation_args)

    def check_if_exists_default_filename(self) -> bool:
        self.arr_path = os.path.join(self.subject_dir, self.default_filename + ".npy")
        return os.path.isfile(self.arr_path)

    def check_if_exists(self, filename:str) -> bool:
        self.arr_path = filename + ".npy"
        return os.path.isfile(self.arr_path)

    def save_arr_default_filename(self, arr:np.array) -> None:
        np.save(self.arr_path, arr) 

    def save_arr(self, arr:np.array, filename:str) -> None:
        self.arr_path = os.path.join(self.subject_dir, filename + ".npy")
        np.save(self.arr_path, arr)