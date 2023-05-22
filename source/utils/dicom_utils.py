import pydicom
import numpy as np
import glob
import scipy.ndimage


# original code from: https://github.com/gallif/_4DCTCostUnrolling with some modifications
def get_filepaths_by_img_num(dir_name, img_num):
    dir_name_for = dir_name + "/IM-00{:02}*.dcm".format(img_num)
    files = [file_name for file_name in glob.glob(dir_name_for, recursive=False)]
    return files
   

# original code from: https://github.com/gallif/_4DCTCostUnrolling with some modifications
def images_to_3d_arr(dir_name, img_num):#, plot=False):
    file_paths = get_filepaths_by_img_num(dir_name, img_num)
    slices = _get_dicom_slices(file_paths)

    # assuming all slices have the same pixel aspects
    pixel_spacing = slices[0].PixelSpacing
    slice_thickness = slices[0].SliceThickness

    img3d, voxel_size = _slices_to_3d_arr(slices, pixel_spacing, slice_thickness)

    img3d_scaled = scipy.ndimage.zoom(img3d, voxel_size)

    return img3d_scaled #TODO create config with voxel_size

def _slices_to_3d_arr(slices, pixel_spacing, slice_thickness):
    voxel_size = np.ndarray(shape=3, buffer=np.array([pixel_spacing[0], pixel_spacing[1], slice_thickness]), dtype=float) 
    img_shape = [*slices[0].pixel_array.shape, len(slices)]
    img3d = np.zeros(img_shape)

    for i, s in enumerate(slices):
        img3d[:, :, i] = s.pixel_array
    return img3d, voxel_size

def get_voxel_size(slice_path): #assuming all sloces has same voxel dim
    slice = pydicom.dcmread(slice_path)
    pixel_spacing = slice.PixelSpacing
    slice_thickness = slice.SliceThickness
    voxel_size = np.ndarray(shape=3, buffer=np.array([pixel_spacing[0], pixel_spacing[1], slice_thickness]), dtype=float) 
    return voxel_size


def _get_dicom_slices(file_paths):
    files = [pydicom.dcmread(file_path) for file_path in file_paths]
    slices = [slice for slice in files if hasattr(slice, "SliceLocation")]
    slices = sorted(slices, key=lambda sl: sl.SliceLocation)
    return slices 