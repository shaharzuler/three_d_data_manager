from dataclasses import dataclass

@dataclass
class MeshSmoothingCreationArgs:
    marching_cubes_step_size: int
    # convexify_method = None #TODO add convex decomposition