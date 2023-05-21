#todo make dataclass
from ast import List


class FilePaths:
    def __init__(self) -> None:
        self.dicom_dir:str = None
        self.dicom_file_paths:List = None
        self.xyz_arr:str = None
        # self.zxy_arr:str = None
        self.two_d_x:str = None
        self.two_d_y:str = None
        self.two_d_z:str = None
        self.two_d:str = None
        self.mesh_raw:str = None
        self.pcd_raw:str = None
