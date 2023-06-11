from dataclasses import dataclass

@dataclass
class LBOCreationArgs:
    num_LBOs: int
    is_point_cloud: bool
    mesh_path: str
    orig_mesh_name: str
    use_torch: bool = True # use_torch=False function is buggy
