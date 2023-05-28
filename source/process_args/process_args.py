from dataclasses import dataclass
import datetime
import os

# class Logger:
#     def __init__(self, outputs_dir) -> None:
#         self.outputs_dir = outputs_dir
#         self.logger_file_path = os.path.join(self.outputs_dir, "log.txt")
#         with open(self.logger_file_path, "w") as f:
#             f.write(f'Logger {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n\n')

#     def log(self, row, verbose):
#         with open(self.logger_file_path, "a") as f:
#             f.write(f"{row}\n")
#         if verbose:
#             print(row)

# LOGGER = Logger(OUTPUTS_DIR)
# def log(row:str, verbose=True):
#     LOGGER.log(row, verbose)


# @dataclass
# class Logable:
#     def __setattr__(self, name, value):
#         super().__setattr__(name, value)
#         log(f"{self.__doc__.split('(')[0]}.{name}: {value}")

@dataclass
class VoxelSmoothingArgs:#(Logable):
    opening_footprint_radius: int
    fill_holes_Area_threshold: int
    closing_to_opening_ratio: float
    # show: bool = False

@dataclass
class MeshSmoothingArgs:#(Logable):
    marching_cubes_step_size: int
    # convexify_method = None #TODO
    

@dataclass
class LBOArgs:#(Logable):
    num_LBOs: int
    is_point_cloud: bool
    max_smooth_lbo_mesh_visualization: int
    use_torch: bool = True # use_torch=False function is buggy


@dataclass
class VoxelizingArgs:
    mesh_path: str