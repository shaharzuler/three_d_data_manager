from three_d_data_manager.source.data_creators.convex_mesh_data_creator import ConvexMeshDataCreator
from three_d_data_manager.source.data_creators.dicom_data_creator import DicomDataCreator
from three_d_data_manager.source.data_creators.h5_data_creator import H5DataCreator
from three_d_data_manager.source.data_creators.lbos_data_creator import LBOsDataCreator
from three_d_data_manager.source.data_creators.mesh_data_creator import MeshDataCreator
from three_d_data_manager.source.data_creators.smooth_lbo_mesh_data_creator import SmoothLBOMeshDataCreator
from three_d_data_manager.source.data_creators.voxelized_mesh_data_creator import VoxelizedMeshDataCreator
from three_d_data_manager.source.data_creators.xyz_arr_data_creator import XYZArrDataCreator
from three_d_data_manager.source.data_creators.xyz_voxels_mask_data_creator import XYZVoxelsMaskDataCreator
from three_d_data_manager.source.data_creators.smooth_voxels_mask_data_creator import SmoothVoxelsMaskDataCreator
from three_d_data_manager.source.data_creators.zxy_voxels_mask_data_creator import ZXYVoxelsMaskDataCreator
from three_d_data_manager.source.data_creators.two_d_vis_data_creator import TwoDVisDataCreator
from three_d_data_manager.source.data_creators.three_d_vis_data_creator import ThreeDVisDataCreator

from three_d_data_manager.source.dataset import Dataset
from three_d_data_manager.source.process_args.h5_dataset_creation_args import H5DatasetCreationArgs
from three_d_data_manager.source.process_args.mesh_smoothing_creation_args import MeshSmoothingCreationArgs
from three_d_data_manager.source.process_args.smooth_mesh_creation_args import SmoothMeshCreationArgs
from three_d_data_manager.source.process_args.voxel_smoothing_creation_args import VoxelSmoothingCreationArgs
from three_d_data_manager.source.process_args.lbo_creation_args import LBOCreationArgs
from three_d_data_manager.source.process_args.voxelizing_creation_args import VoxelizingCreationArgs
from three_d_data_manager.source.process_args.two_d_vis_creation_args import TwoDVisualizationCreationArgs
from three_d_data_manager.source.process_args.three_d_vis_creation_args import ThreeDVisualizationCreationArgs



# create dataset
dataset_target_path = "path/to/target/dataset"
timesteps = ("18", "28")
zxy_voxels_mask_arr_paths = ("path/to/arr/mask/folder18", "path/to/arr/mask/folder28")

dataset = Dataset(target_root_dir=dataset_target_path)
for sample_name, zxy_voxels_mask_arr_path in zip(timesteps, zxy_voxels_mask_arr_paths):
    # add dicom
    dicom_path = "path/to/dicom/folder" 
    dicom_data_creator = DicomDataCreator(source_path=dicom_path, sample_name=sample_name, hirarchy_levels=2) #2 hirarchy levels allowed you to hold 2 versions of the same data sample. for example, a mask of manual tagging and a mask warped from another timestep.
    dataset.add_sample(dicom_data_creator)

    # create np arrays from dicom
    xyz_arr_data_creator = XYZArrDataCreator(source_path=None, sample_name=sample_name, hirarchy_levels=2)
    dataset.add_sample(xyz_arr_data_creator)

    # add segmentation mask
    zxy_voxels_mask_data_creator = ZXYVoxelsMaskDataCreator(source_path=zxy_voxels_mask_arr_path, sample_name=sample_name, hirarchy_levels=2)
    dataset.add_sample(zxy_voxels_mask_data_creator)

    # zxy to xyz
    xyz_voxels_mask_data_creator = XYZVoxelsMaskDataCreator(source_path=None, sample_name=sample_name, hirarchy_levels=2)
    dataset.add_sample(xyz_voxels_mask_data_creator) 
    # smooth mask with voxels methods
    voxel_smoothing_args = VoxelSmoothingCreationArgs(opening_footprint_radius=7, fill_holes_Area_threshold=1000, closing_to_opening_ratio=0.85)
    smooth_voxel_mask_data_creator = SmoothVoxelsMaskDataCreator(source_path=None, sample_name=sample_name, hirarchy_levels=2, creation_args=voxel_smoothing_args)
    dataset.add_sample(smooth_voxel_mask_data_creator)

    # create mesh from voxels
    mesh_creation_args = MeshSmoothingCreationArgs(marching_cubes_step_size=1) 
    mesh_data_creator = MeshDataCreator(source_path=None, sample_name=sample_name, hirarchy_levels=2, creation_args=mesh_creation_args)
    dataset.add_sample(mesh_data_creator) 

    # check the affect of re-voxelizing the original mesh (created with marching cubes from original xyz mask):
    mesh_voxelizing_args = VoxelizingCreationArgs(mesh_path=dataset.file_paths.mesh[sample_name])
    voxelized_mesh_data_creator = VoxelizedMeshDataCreator(source_path=None, sample_name=sample_name, hirarchy_levels=2, creation_args=mesh_voxelizing_args)
    dataset.add_sample(voxelized_mesh_data_creator)

    # create lbos from mesh
    lbo_creation_args = LBOCreationArgs(num_LBOs=300, is_point_cloud=False, mesh_path=dataset.file_paths.mesh[sample_name], orig_mesh_name="mesh", use_torch=True)
    lbos_data_creator = LBOsDataCreator(source_path=None, sample_name=sample_name, hirarchy_levels=2, creation_args=lbo_creation_args)
    dataset.add_sample(lbos_data_creator)

    #create mesh h5 dataset
    h5_dataset_creation_args = H5DatasetCreationArgs(orig_name="mesh", override=True)
    h5_dataset_data_creator = H5DataCreator(source_path=None, sample_name=sample_name, hirarchy_levels=2, creation_args=h5_dataset_creation_args)
    dataset.add_sample(h5_dataset_data_creator)

    # smooth mesh with lbos
    smooth_mesh_creation_args = SmoothMeshCreationArgs(lbos_path=dataset.file_paths.mesh_lbo_data[sample_name])
    smooth_lbo_mesh_data_creator = SmoothLBOMeshDataCreator(source_path=None, sample_name=sample_name, hirarchy_levels=2, creation_args=smooth_mesh_creation_args)
    dataset.add_sample(smooth_lbo_mesh_data_creator)

    # voxelize smooth mesh back
    smooth_mesh_voxelizing_args = VoxelizingCreationArgs(mesh_path=dataset.file_paths.mesh_smooth[sample_name])
    voxelized_smooth_mesh_data_creator = VoxelizedMeshDataCreator(source_path=None, sample_name=sample_name, hirarchy_levels=2, creation_args=smooth_mesh_voxelizing_args)
    dataset.add_sample(voxelized_smooth_mesh_data_creator)

    # create lbos from smooth mesh
    smooth_lbo_creation_args = LBOCreationArgs(num_LBOs=300, is_point_cloud=False, mesh_path=dataset.file_paths.mesh_smooth[sample_name], orig_mesh_name="mesh_smooth", use_torch=True)
    smooth_lbos_data_creator = LBOsDataCreator(source_path=None, sample_name=sample_name, hirarchy_levels=2, creation_args=smooth_lbo_creation_args)
    dataset.add_sample(smooth_lbos_data_creator)

    # create smooth mesh h5 dataset
    smooth_h5_dataset_creation_args = H5DatasetCreationArgs(orig_name="mesh_smooth", override=True)
    smooth_h5_dataset_data_creator = H5DataCreator(source_path=None, sample_name=sample_name, hirarchy_levels=2, creation_args=smooth_h5_dataset_creation_args)
    dataset.add_sample(smooth_h5_dataset_data_creator)

    # create convex hall of smooth mesh
    convex_mesh_data_creator = ConvexMeshDataCreator(source_path=None, sample_name=sample_name, hirarchy_levels=2)
    dataset.add_sample(convex_mesh_data_creator)
    convex_mesh_voxelizing_args = VoxelizingCreationArgs(mesh_path=dataset.file_paths.mesh_convex[sample_name])
    voxelized_convex_mesh_data_creator = VoxelizedMeshDataCreator(source_path=None, sample_name=sample_name, hirarchy_levels=2, creation_args=convex_mesh_voxelizing_args)
    dataset.add_sample(voxelized_convex_mesh_data_creator)

    # create lbos from convex mesh
    convex_lbo_creation_args = LBOCreationArgs(num_LBOs=300, is_point_cloud=False, mesh_path=dataset.file_paths.mesh_convex[sample_name], orig_mesh_name="mesh_convex", use_torch=True)
    convex_lbos_data_creator = LBOsDataCreator(source_path=None, sample_name=sample_name, hirarchy_levels=2, creation_args=convex_lbo_creation_args)
    dataset.add_sample(convex_lbos_data_creator)

    #create convex mesh h5 dataset
    convex_h5_dataset_creation_args = H5DatasetCreationArgs(orig_name="mesh_convex", override=True)
    convex_h5_dataset_data_creator = H5DataCreator(source_path=None, sample_name=sample_name, hirarchy_levels=2, creation_args=convex_h5_dataset_creation_args)
    dataset.add_sample(convex_h5_dataset_data_creator)

    dataset.save_file_paths()

    # getters:
    xyz_voxels_mask_arr = dataset.get_xyz_voxels_mask(sample_name)
    xyz_voxels_mask_smooth = dataset.get_smooth_voxels_mask(sample_name)
    xyz_arr = dataset.get_xyz_arr(sample_name)
    zxy_arr = dataset.get_zxy_arr(sample_name)
    mesh_verts, mesh_faces = dataset.get_mesh(sample_name)
    smooth_mesh_verts, smooth_mesh_faces = dataset.get_smooth_mesh(sample_name)
    lbo_eigenvectors, lbo_eigenvalues, lbo_area_weights = dataset.get_mesh_lbo_data(sample_name)
    convex_verts, convex_faces =  dataset.get_convex_mesh(sample_name)
    voxelized_smooth_mesh = dataset.get_voxelized_smooth_mesh(sample_name)
    voxelized_convex_mesh = dataset.get_voxelized_convex_mesh(sample_name)

    # create single sample level visualizations
    two_d_visualization_args = TwoDVisualizationCreationArgs()
    two_d_visualization_data_creator = TwoDVisDataCreator(source_path=None, sample_name=sample_name, hirarchy_levels=2, creation_args=two_d_visualization_args)
    dataset.visualize_existing_data_sections(two_d_visualization_data_creator)

    three_d_vizualisation_args = ThreeDVisualizationCreationArgs(max_smooth_lbo_mesh_visualization=6)
    three_d_visualization_data_creator = ThreeDVisDataCreator(source_path=None, sample_name=sample_name, hirarchy_levels=2, creation_args=three_d_vizualisation_args)
    dataset.visualize_existing_data_3d(three_d_visualization_data_creator)









