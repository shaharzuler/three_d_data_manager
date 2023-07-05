from dataclasses import dataclass

@dataclass
class PointCloudCreationArgs:
    num_points: int
    orig_mesh_name:str
    mesh_path: str