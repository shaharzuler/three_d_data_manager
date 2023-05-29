import numpy as np
from skimage import measure
import pyvista as pv
from typing import Tuple
import math
import numpy as np
import trimesh

def voxels_mask_to_mesh(voxels_mask:np.array, marching_cubes_step_size:int):
    verts, faces, normals, values = measure.marching_cubes(
        voxels_mask,
        gradient_direction="descent",
        step_size=marching_cubes_step_size, 
        allow_degenerate=False) 
    return verts, faces, normals, values



class Mesh2VoxelsConvertor:
    def __init__(self, mesh_path, original_shape):
        self.mesh = pv.read(mesh_path) 
        pv_voxels = pv.voxelize(self.mesh, density=1, check_surface=False)
        tight_sparse_ind = np.round((pv_voxels.cell_centers().points - np.min(pv_voxels.cell_centers().points, 0)))
        tight_dense_voxels = (trimesh.voxel.ops.sparse_to_matrix(tight_sparse_ind )).astype(bool)
        x_pad_bef, y_pad_bef, z_pad_bef, x_pad_aft, y_pad_aft, z_pad_aft = self.get_padding_values_for_voxels(self.mesh, original_shape, tight_dense_voxels.shape)
        self.padded_voxelized = np.pad(tight_dense_voxels, ((x_pad_bef, x_pad_aft), (y_pad_bef, y_pad_aft), (z_pad_bef, z_pad_aft)))

    @staticmethod
    def get_padding_values_for_voxels(mesh:pv.PolyData, original_shape:Tuple[int], unpadded_shape:Tuple[int]):
        pad_before = np.round(abs(np.min(mesh.points, axis=0))).astype(int)
        pad_after = np.array(original_shape) - np.array(unpadded_shape) - pad_before
        return *pad_before, *pad_after