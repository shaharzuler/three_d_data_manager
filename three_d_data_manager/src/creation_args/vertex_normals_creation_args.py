from dataclasses import dataclass

@dataclass
class VertexNormalsCreationArgs:
    k_nn_for_normals_calc: int
    orig_geometry_name:str
    geometry_path: str