from dataclasses import dataclass

@dataclass
class H5DatasetCreationArgs:
    orig_name: str
    is_point_cloud: bool
    override: bool = False
    

    