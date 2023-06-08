# 3D Data Manager Library Guide

The **three_d_data_manager** is a Python library designed to manage and process 3D data samples. It offers various functionalities for creating and processing datasets from 3D scans in DICOM format and their corresponding 3D boolean segmentation arrays. The library provides tools to convert segmentation arrays to multiple formats. Below is a detailed guide on how to use the library with example code snippets.
## Installation

To install the **three_d_data_manager**, use the following command:
```b
pip install three_d_data_manager
```

## usage
To begin using the library, follow the steps outlined below.

### Initialize a new dataset

To create a new dataset, initialize the ```Dataset``` class with the target root directory where the dataset will be stored:
```python
dataset = Dataset(target_root_dir="target/path/to/dataset")
```
### Add DICOM data and create 3D numpy arrays

To add DICOM data and create 3D numpy arrays, use the ```DicomDataCreator``` and ```XYZArrDataCreator``` classes. First, create directories for multiple sample names (DICOM timesteps) and specify the DICOM files' source path:<br>

```python
sample_name = "18"
dicom_path = "path/to/dicom/folder" 

dicom_data_creator = DicomDataCreator(
    source_path=dicom_path, 
    sample_name=sample_name, 
    hirarchy_levels=2
) 

dataset.add_sample(dicom_data_creator)
```
The DICOM files will be copied to the following directory: <br>
`target/path/to/dataset/18/orig/DICOM`<br>
Defining 2 <b>hirarchy levels</b> will allow you to hold 2 versions of the same data sample. For example, a mask of manual segmentation and a mask warped from another timestep.
<br/><br/>
To create 3D numpy arrays from separate 2D DICOM files, use the ```XYZArrDataCreator``` class:
```python
xyz_arr_data_creator = XYZArrDataCreator(
    source_path=None, 
    sample_name=sample_name, 
    hirarchy_levels=2
)

dataset.add_sample(xyz_arr_data_creator)
```
The resulting 3D numpy array will be saved to: <br>
`target/path/to/dataset/18/orig/voxels/xyz_arr_raw.npy`
### Add segmentation masks

To add segmentation binary masks in ZXY format and create masks in XYZ format, use the ```ZXYVoxelsMaskDataCreator``` and ```XYZVoxelsMaskDataCreator``` classes:
```python
zxy_voxels_mask_data_creator = ZXYVoxelsMaskDataCreator(
    source_path=zxy_voxels_mask_arr_path, 
    sample_name=sample_name, 
    hirarchy_levels=2
)

dataset.add_sample(zxy_voxels_mask_data_creator)

xyz_voxels_mask_data_creator = XYZVoxelsMaskDataCreator(
    source_path=None, 
    sample_name=sample_name, 
    hirarchy_levels=2)

dataset.add_sample(xyz_voxels_mask_data_creator) 
```
The following numpy arrays will be created:<br>
`target/path/to/dataset/18/orig/voxels/zxy_voxels_mask_raw.npy` <br>
`target/path/to/dataset/18/orig/voxels/xyz_voxels_mask_raw.npy` <br>
### Perform voxel smoothing

To close small holes and smooth the mask in voxel space using morphological opening and closing operations, utilize the ```SmoothVoxelsMaskDataCreator``` class. Define the smoothing parameters with a ```VoxelSmoothingCreationArgs``` instance:
```python
voxel_smoothing_args = VoxelSmoothingCreationArgs(
    opening_footprint_radius=7, 
    fill_holes_Area_threshold=1000, 
    closing_to_opening_ratio=0.85
)

smooth_voxel_mask_data_creator = SmoothVoxelsMaskDataCreator(
    source_path=None, 
    sample_name=sample_name, 
    hirarchy_levels=2
)

dataset.add_sample(smooth_voxel_mask_data_creator, voxel_smoothing_args)
```
The resulting numpy 3D mask will be saved to: <br>
`target/path/to/dataset/18/orig/voxels/xyz_voxels_mask_smooth.npy` <br>
A configuration file documenting the smoothening parameters will be saved to:<br>
`target/path/to/dataset/18/orig/voxels/xyz_voxels_mask_smooth_config.json` <br>
### Create a mesh from the voxel mask

To create a mesh from the voxel mask, use the ```MeshDataCreator``` class. You can also specify mesh creation arguments using the ```MeshSmoothingCreationArgs``` class:
```python
mesh_creation_args = MeshSmoothingCreationArgs(marching_cubes_step_size=1) 

mesh_data_creator = MeshDataCreator(
    source_path=None, 
    sample_name=sample_name, 
    hirarchy_levels=2
)

dataset.add_sample(mesh_data_creator, mesh_creation_args) 
```
An off file of the mesh will be saved to: <br>
`target/path/to/dataset/18/orig/meshes/mesh.off` <br>
A configuration file documenting the mesh creation arguments will be saved to:<br>
`target/path/to/dataset/18/orig/meshes/mesh_config.json` <br>
### Calculate and store Laplacian Beltrami Operators (LBOs)
To calculate and store LBOs of the mesh file, use the ```LBOsDataCreator``` class. Specify the LBO creation arguments with an instance of ```LBOCreationArgs```:
```python
lbo_creation_args = LBOCreationArgs(
    num_LBOs=300, 
    is_point_cloud=False, 
    mesh_path=dataset.file_paths.mesh[sample_name], 
    orig_mesh_name="mesh", 
    use_torch=True
)

lbos_data_creator = LBOsDataCreator(
    source_path=None, 
    sample_name=sample_name, 
    hirarchy_levels=2
)

dataset.add_sample(lbos_data_creator, lbo_creation_args)

```
LBO eigenvectors, eigenvalues and area weights will be saved to: <br>
`target/path/to/dataset/18/orig/lbos/mesh_lbo_data.npz` <br>
A configuration file documenting the mesh creation arguments will be saved to: <br>
`target/path/to/dataset/18/orig/lbos/mesh_lbo_data_config.json` <br>
### Create an H5 format dataset
To create an H5 format dataset, use the ```H5DataCreator``` class. Specify the dataset creation arguments with an instance of ```H5DatasetCreationArgs```:
```python
h5_dataset_creation_args = H5DatasetCreationArgs(
    orig_name="mesh", 
    override=True
)

h5_dataset_data_creator = H5DataCreator(
    source_path=None, 
    sample_name=sample_name, 
    hirarchy_levels=2
)

dataset.add_sample(h5_dataset_data_creator, h5_dataset_creation_args)
```
The H5 file will be saved to: <br>
`target/path/to/dataset/18/orig/h5_datasets/mesh_dataset.h5df` <br>
### Smooth the mesh using LBOs
To smooth the mesh using LBOs, utilize the ```SmoothLBOMeshDataCreator``` class. Specify the smoothing arguments with an instance of ```SmoothMeshCreationArgs```:
```python
smooth_mesh_creation_args = SmoothMeshCreationArgs(
    lbos_path=dataset.file_paths.mesh_lbo_data[sample_name]
)

smooth_lbo_mesh_data_creator = SmoothLBOMeshDataCreator(
    source_path=None, 
    sample_name=sample_name, 
    hirarchy_levels=2
)

dataset.add_sample(smooth_lbo_mesh_data_creator, smooth_mesh_creation_args)
```
The smoothed mesh will be saved to: <br>
`target/path/to/dataset/18/orig/meshes/smooth_mesh.off` <br>
A configuration file documenting the mesh smoothening arguments will be saved to: <br>
`target/path/to/dataset/18/orig/meshes/smooth_mesh_config.json` <br>
### Voxelize the smooth mesh
To voxelize the smooth mesh, use the ```VoxelizedMeshDataCreator``` class. Specify the voxelization arguments with an instance of ```VoxelizingCreationArgs```:
```python
smooth_mesh_voxelizing_args = VoxelizingCreationArgs(
    mesh_path=dataset.file_paths.mesh_smooth[sample_name]
)

voxelized_smooth_mesh_data_creator = VoxelizedMeshDataCreator(
    source_path=None, 
    sample_name=sample_name, 
    hirarchy_levels=2
)

dataset.add_sample(voxelized_smooth_mesh_data_creator, smooth_mesh_voxelizing_args)
```
The voxelized version of the mesh, in the dimension of the original voxel mask, will be saved to: <br>
`target/path/to/dataset/18/orig/voxels/xyz_smooth_mesh_voxelized.npy` <br>
### Calculate LBOs for the smoothed mesh
```python
smooth_lbo_creation_args = LBOCreationArgs(
    num_LBOs=300, 
    is_point_cloud=False, 
    mesh_path=dataset.file_paths.mesh_smooth[sample_name], 
    orig_mesh_name="mesh_smooth", 
    use_torch=True
)

smooth_lbos_data_creator = LBOsDataCreator(
    source_path=None,
    sample_name=sample_name, 
    hirarchy_levels=2
)

dataset.add_sample(smooth_lbos_data_creator, smooth_lbo_creation_args)
```

LBO eigenvectors, eigenvalues and area weights will be saved to: <br>
`target/path/to/dataset/18/orig/lbos/mesh_smooth_lbo_data.npz`<br>
A configuration file documenting the smoothening arguments  will be saved to: <br>
`target/path/to/dataset/18/orig/lbos/mesh_smooth_lbo_data_config.json` <br>
### Create a convex hull of the smooth mesh and voxelize it
```python
convex_mesh_data_creator = ConvexMeshDataCreator(
    source_path=None, 
    sample_name=sample_name, 
    hirarchy_levels=2
)

dataset.add_sample(convex_mesh_data_creator)

convex_mesh_voxelizing_args = VoxelizingCreationArgs(
    mesh_path=dataset.file_paths.mesh_convex[sample_name]
)

voxelized_convex_mesh_data_creator = VoxelizedMeshDataCreator(
    source_path=None, 
    sample_name=sample_name, 
    hirarchy_levels=2)

dataset.add_sample(voxelized_convex_mesh_data_creator, convex_mesh_voxelizing_args)
```
The convex hull mesh will be saved to: <br>
`target/path/to/dataset/18/orig/meshes/convex_mesh.off` <br>
The voxelized version of the convex hull mesh will be saved to: <br>
`target/path/to/dataset/18/orig/voxels/xyz_convex_mesh_voxelized.npy` <br>

### Persistency
To save the file paths of created data samples, in order to optionally start a future Dataset instance with, use the ```save_file_paths``` method of the ```Dataset``` class.


### Data visualization
To visualize the data samples, use the ```TwoDVisDataCreator``` and ```ThreeDVisDataCreator``` classes.
```python
two_d_visualization_args = TwoDVisualizationCreationArgs()

two_d_visualization_data_creator = TwoDVisDataCreator(
    source_path=None, 
    sample_name=sample_name, 
    hirarchy_levels=2
)

dataset.visualize_existing_data_sections(two_d_visualization_data_creator, two_d_visualization_args)

three_d_vizualisation_args = ThreeDVisualizationCreationArgs(max_smooth_lbo_mesh_visualization=6)

three_d_visualization_data_creator = ThreeDVisDataCreator(
    source_path=None, 
    sample_name=sample_name, 
    hirarchy_levels=2
)

dataset.visualize_existing_data_3d(three_d_visualization_data_creator, three_d_vizualisation_args)

```
3 main sections of the scan without and with all versions of segmentation masks will be saved to: <br>
`target/path/to/dataset/18/orig/2d_sections_visualization`<br>
The mesh and first 6 LBOs will be saved to: <br>
`target/path/to/dataset/18/orig/3d_visualization`<br>
### Getters:
you can use the getters to load the files from the dataset:
```python
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
```

## Conclusion
The final result for timestep 18 will be as follows:
<pre>
.
└── orig
    ├── 2d_sections_visualization
    │   ├── 2d_sections_config.json
    │   ├── n0_raw_mask_sections.jpg
    │   ├── n1_smooth_by_voxels_mask_sections.jpg
    │   ├── n2_mesh_mask_sections.jpg
    │   ├── n3_smooth_by_lbo_mask_sections.jpg
    │   ├── n4_convex_mask_sections.jpg
    │   ├── scan_2d_sections.jpg
    │   └── section_contours.jpg
    ├── 3d_visualization
    │   ├── 3d_smooth_0.html
    │   ├── 3d_smooth_0.json
    │   ├── 3d_smooth_0.png
    │   ├── 3d_smooth_1.html
    │   ├── 3d_smooth_1.json
    │   ├── 3d_smooth_1.png
    │   ├── 3d_smooth_2.html
    │   ├── 3d_smooth_2.json
    │   ├── 3d_smooth_2.png
    │   ├── 3d_smooth_3.html
    │   ├── 3d_smooth_3.json
    │   ├── 3d_smooth_3.png
    │   ├── 3d_smooth_4.html
    │   ├── 3d_smooth_4.json
    │   ├── 3d_smooth_4.png
    │   ├── 3d_smooth_5.html
    │   ├── 3d_smooth_5.json
    │   ├── 3d_smooth_5.png
    │   ├── 3d_smooth_config.json
    │   ├── 3d_smooth_lbo_grid.png
    │   ├── smooth_clean_mesh_0.html
    │   ├── smooth_clean_mesh_0.json
    │   ├── smooth_clean_mesh_0.png
    │   └── smooth_clean_mesh_lbo_grid.png
    ├── DICOM
    │   ├── IM-0018-0911-0001.dcm
    │   ├── IM-0018-0912-0001.dcm
    │   ├── IM-0018-0913-0001.dcm
    │   ├── IM-0018-0914-0001.dcm
    │   ├── IM-0018-0915-0001.dcm
    │   ├── IM-0018-0916-0001.dcm
    │   ├── IM-0018-0917-0001.dcm
    │   ├── IM-0018-0918-0001.dcm
    │   ├── IM-0018-0919-0001.dcm
    │   ├── IM-0018-0920-0001.dcm
    │   ├── IM-0018-0921-0001.dcm
    │   ├── IM-0018-0922-0001.dcm
    │   ├── IM-0018-0923-0001.dcm
    │   ├── IM-0018-0924-0001.dcm
    │   ├── IM-0018-0925-0001.dcm
    │   ├── IM-0018-0926-0001.dcm
    │   ├── IM-0018-0927-0001.dcm
    │   ├── IM-0018-0928-0001.dcm
    │   ├── IM-0018-0929-0001.dcm
    │   ├── IM-0018-0930-0001.dcm
    │   ├── IM-0018-0931-0001.dcm
    │   ├── IM-0018-0932-0001.dcm
    │   ├── IM-0018-0933-0001.dcm
    │   ├── IM-0018-0934-0001.dcm
    │   ├── IM-0018-0935-0001.dcm
    │   ├── IM-0018-0936-0001.dcm
    │   ├── IM-0018-0937-0001.dcm
    │   ├── IM-0018-0938-0001.dcm
    │   ├── IM-0018-0939-0001.dcm
    │   ├── IM-0018-0940-0001.dcm
    │   ├── IM-0018-0941-0001.dcm
    │   ├── IM-0018-0942-0001.dcm
    │   ├── IM-0018-0943-0001.dcm
    │   ├── IM-0018-0944-0001.dcm
    │   ├── IM-0018-0945-0001.dcm
    │   ├── IM-0018-0946-0001.dcm
    │   ├── IM-0018-0947-0001.dcm
    │   ├── IM-0018-0948-0001.dcm
    │   ├── IM-0018-0949-0001.dcm
    │   ├── IM-0018-0950-0001.dcm
    │   ├── IM-0018-0951-0001.dcm
    │   ├── IM-0018-0952-0001.dcm
    │   ├── IM-0018-0953-0001.dcm
    │   ├── IM-0018-0954-0001.dcm
    │   ├── IM-0018-0955-0001.dcm
    │   ├── IM-0018-0956-0001.dcm
    │   ├── IM-0018-0957-0001.dcm
    │   ├── IM-0018-0958-0001.dcm
    │   ├── IM-0018-0959-0001.dcm
    │   ├── IM-0018-0960-0001.dcm
    │   ├── IM-0018-0961-0001.dcm
    │   ├── IM-0018-0962-0001.dcm
    │   ├── IM-0018-0963-0001.dcm
    │   ├── IM-0018-0964-0001.dcm
    │   ├── IM-0018-0965-0001.dcm
    │   ├── IM-0018-0966-0001.dcm
    │   ├── IM-0018-0967-0001.dcm
    │   ├── IM-0018-0968-0001.dcm
    │   ├── IM-0018-0969-0001.dcm
    │   ├── IM-0018-0970-0001.dcm
    │   ├── IM-0018-0971-0001.dcm
    │   ├── IM-0018-0972-0001.dcm
    │   ├── IM-0018-0973-0001.dcm
    │   ├── IM-0018-0974-0001.dcm
    │   ├── IM-0018-0975-0001.dcm
    │   ├── IM-0018-0976-0001.dcm
    │   ├── IM-0018-0977-0001.dcm
    │   ├── IM-0018-0978-0001.dcm
    │   ├── IM-0018-0979-0001.dcm
    │   ├── IM-0018-0980-0001.dcm
    │   ├── IM-0018-0981-0001.dcm
    │   ├── IM-0018-0982-0001.dcm
    │   ├── IM-0018-0983-0001.dcm
    │   ├── IM-0018-0984-0001.dcm
    │   ├── IM-0018-0985-0001.dcm
    │   ├── IM-0018-0986-0001.dcm
    │   ├── IM-0018-0987-0001.dcm
    │   ├── IM-0018-0988-0001.dcm
    │   ├── IM-0018-0989-0001.dcm
    │   ├── IM-0018-0990-0001.dcm
    │   ├── IM-0018-0991-0001.dcm
    │   ├── IM-0018-0992-0001.dcm
    │   ├── IM-0018-0993-0001.dcm
    │   ├── IM-0018-0994-0001.dcm
    │   ├── IM-0018-0995-0001.dcm
    │   ├── IM-0018-0996-0001.dcm
    │   ├── IM-0018-0997-0001.dcm
    │   ├── IM-0018-0998-0001.dcm
    │   ├── IM-0018-0999-0001.dcm
    │   ├── IM-0018-1000-0001.dcm
    │   └── IM-0018-1001-0001.dcm
    ├── h5_datasets
    │   ├── mesh_convex_dataset_config.json
    │   ├── mesh_convex_dataset.hdf5
    │   ├── mesh_dataset_config.json
    │   ├── mesh_dataset.hdf5
    │   ├── mesh_smooth_dataset_config.json
    │   └── mesh_smooth_dataset.hdf5
    ├── lbos
    │   ├── mesh_convex_lbo_data_config.json
    │   ├── mesh_convex_lbo_data.npz
    │   ├── mesh_lbo_data_config.json
    │   ├── mesh_lbo_data.npz
    │   ├── mesh_smooth_lbo_data_config.json
    │   └── mesh_smooth_lbo_data.npz
    ├── meshes
    │   ├── convex_mesh.off
    │   ├── mesh_config.json
    │   ├── mesh.off
    │   ├── smooth_mesh_config.json
    │   └── smooth_mesh.off
    └── voxels
        ├── xyz_arr_raw.npy
        ├── xyz_convex_mesh_voxelized.npy
        ├── xyz_mesh_voxelized.npy
        ├── xyz_smooth_mesh_voxelized.npy
        ├── xyz_voxelized_config.json
        ├── xyz_voxels_mask_raw.npy
        ├── xyz_voxels_mask_smooth_config.json
        ├── xyz_voxels_mask_smooth.npy
        └── zxy_voxels_mask_raw.npy
