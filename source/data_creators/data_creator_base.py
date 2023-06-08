import os

from three_d_data_manager.source.file_paths import FilePaths

class DataCreatorBase:
    def __init__(self, source_path:str, sample_name:str, hirarchy_levels:int, override:bool=True, version_name:str="orig") -> None:
        self.source_path: str = source_path # use source_path!= None if you prefer providing the processed file yourself rather than using the library to process it.
        self.sample_name: str = sample_name
        self.hirarchy_levels: int = hirarchy_levels
        self.version_name: str = version_name
        self.default_dirname = "default_subject_dirname"
        self.override: bool = override

    def add_sample(self, target_root_dir:str, creation_args, dataset_attrs:dict[str,str]):
        if self.hirarchy_levels>2:
            self.sample_dir = os.path.join(target_root_dir, self.sample_name, *([self.version_name]*self.hirarchy_levels))
        else: 
            self.sample_dir = os.path.join(target_root_dir, self.sample_name, self.version_name)
        os.makedirs(self.sample_dir, exist_ok=True)

        self.subject_dir = os.path.join(self.sample_dir, self.default_dirname)
        os.makedirs(self.subject_dir, exist_ok=True)

    def add_sample_from_file(self, file, target_root_dir:str, file_paths:FilePaths, creation_args, dataset_attrs:dict[str,str]):
        raise NotImplementedError

    def get_properties(self) -> dict[str, any]:
        return {}

    def check_if_exists(self, filename:str) -> bool:
        return os.path.isfile(filename)
