import trimesh
import numpy as np
import os
import scipy
import pyvista as pv

mgh=False
magix=True

if magix:
    # Measured manually using 3d-slicer:
    img_xmin_default = -130.8
    img_xmax_default = 74.8
    img_ymin_default = 63.2
    img_ymax_default = 268.8
    img_zmin_default = -232.5
    img_zmax_default = -82.5

    out_shape_x = 206
    out_shape_y = 206
    out_shape_z = 152 

    # paths:
    # orig_mesh_dir = r"/home/shahar/data/magix/manual_stl_seg"
    # output_mesh_dir = r"/home/shahar/data/magix/manual_stl_seg_mirror"
    # output_arr_dir = r"/home/shahar/data/magix/manual_np_seg"

    # orig_mesh_dir = r"/home/shahar/data/magix/only_ts_30/manual_stl_seg"
    # output_mesh_dir = r"/home/shahar/data/magix/only_ts_30/manual_stl_seg_mirror"
    # output_arr_dir = r"/home/shahar/data/magix/only_ts_30/manual_np_seg"

    orig_mesh_dir = r"/home/shahar/data/magix/manual_stl_seg"
    output_mesh_dir = r"/home/shahar/data/magix/manual_stl_seg_mirror"
    output_arr_dir = r"/home/shahar/data/magix/manual_np_seg"


elif mgh:
    img_xmin_default = -92.6
    img_xmax_default = 106.6
    img_ymin_default = -305.1
    img_ymax_default = -105.9
    img_zmin_default = -313.5
    img_zmax_default = -178.5

    out_shape_x = 256
    out_shape_y = 256
    out_shape_z = 91

    # paths:
    orig_mesh_dir = r"/home/shahar/projects/pcd_to_mesh/exploration/lv_gdl/mesh_files/lumen_manual_seg_nati_Aug22/raw_from_nati"
    output_mesh_dir = r"/home/shahar/projects/pcd_to_mesh/exploration/lv_gdl/mesh_files/lumen_manual_seg_nati_Aug22/processed_meshes"
    output_arr_dir = r"/home/shahar/projects/pcd_to_mesh/exploration/lv_gdl/manual_seg_npzs_Aug22"



def get_padding_values_for_voxels(mesh, img_xmin, img_xmax, img_ymin, img_ymax, img_zmin, img_zmax):
    mesh_xmin, mesh_ymin, mesh_zmin = np.min(mesh.points, axis=0)
    mesh_xmax, mesh_ymax, mesh_zmax = np.max(mesh.points, axis=0)
    print("mesh min max vals", mesh_xmin, mesh_ymin, mesh_zmin, mesh_xmax, mesh_ymax, mesh_zmax)
    print("img min max vals", img_xmin, img_xmax, img_ymin, img_ymax, img_zmin, img_zmax)

    x_pad_bef = round(abs(mesh_xmin - img_xmin))
    x_pad_aft = round(abs(mesh_xmax - img_xmax))
    y_pad_bef = round(abs(mesh_ymin - img_ymin))
    y_pad_aft = round(abs(mesh_ymax - img_ymax))
    z_pad_bef = round(abs(mesh_zmin - img_zmin))
    z_pad_aft = round(abs(mesh_zmax - img_zmax))
    return x_pad_bef, x_pad_aft, y_pad_bef, y_pad_aft, z_pad_bef, z_pad_aft

if __name__ == "__main__": 
    for filename in os.listdir(orig_mesh_dir):
        print(os.path.join(orig_mesh_dir, filename))
        mesh = pv.read(os.path.join(orig_mesh_dir, filename))
        if mgh:
            # mirror about shifted z:
            mesh.points[:, -1] = mesh.points[:, -1] - img_zmin
            mesh.points[:, -1] = -mesh.points[:, -1]
            mesh.points[:, -1] = mesh.points[:, -1] + img_zmin
            img_xmin, img_xmax = img_xmax_default, img_xmin_default
            img_ymin, img_ymax = img_ymax_default, img_ymin_default
            img_zmin, img_zmax = img_zmax_default, img_zmin_default
        elif magix:
            mesh.points[:, -1] = -mesh.points[:, -1]
            img_xmin, img_xmax = -img_xmax_default, -img_xmin_default
            img_ymin, img_ymax = -img_ymax_default, -img_ymin_default
            img_zmin, img_zmax = -img_zmax_default, -img_zmin_default

        mesh.save(os.path.join(output_mesh_dir, filename))

        pv_voxels = pv.voxelize(mesh, density=1, check_surface=False)

        tight_sparse_ind = np.round((pv_voxels.cell_centers().points-np.min(pv_voxels.cell_centers().points,0)))
        tight_dense_voxels = trimesh.voxel.ops.sparse_to_matrix(tight_sparse_ind )
        print(tight_dense_voxels.shape, "mm before padding")
        x_pad_bef, x_pad_aft, y_pad_bef, y_pad_aft, z_pad_bef, z_pad_aft = get_padding_values_for_voxels(mesh, img_xmin, img_xmax, img_ymin, img_ymax, img_zmin, img_zmax)
        print("pad vals", x_pad_bef, x_pad_aft, y_pad_bef, y_pad_aft, z_pad_bef, z_pad_aft)

        padded_voxelized = np.pad(tight_dense_voxels, ((x_pad_bef, x_pad_aft), (y_pad_bef, y_pad_aft), (z_pad_bef, z_pad_aft)))
        print(padded_voxelized.shape, "mm")
        # adjust voxelized dimensions to dims expected by the postprocessing class:
        rolled_voxelized = np.rollaxis(padded_voxelized, -1, 0)
        rolled_voxelized = np.swapaxes(rolled_voxelized, 1, 2)
        voxelized = scipy.ndimage.zoom(
            rolled_voxelized, 
            (out_shape_z / rolled_voxelized.shape[0], out_shape_x / rolled_voxelized.shape[1], out_shape_y / rolled_voxelized.shape[2]),
            order=0, mode="nearest") 
        print(voxelized.shape,"voxels")
        print(voxelized.sum(), "sum")
        timestep = filename.replace(".stl","").replace("seg","")
        np.save(os.path.join(output_arr_dir, timestep+"_to_"+timestep+".npy"), voxelized)


