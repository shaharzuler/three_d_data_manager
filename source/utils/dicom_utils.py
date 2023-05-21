import pydicom
import numpy as np
import glob
import scipy.ndimage

def get_filepaths_by_img_num(dir_name, img_num):
    dir_name_for = dir_name + "/IM-00{:02}*.dcm".format(img_num)
    files = [file_name for file_name in glob.glob(dir_name_for, recursive=False)]
    return files

    # files = []
    # print("glob: {}".format(dir_name_for))
    
    # for fname in glob.glob(dir_name_for, recursive=False):
    #     files.append(fname)
    

# original code from: https://github.com/gallif/_4DCTCostUnrolling with some modifications
def images_to_3d_arr(dir_name, img_num):#, plot=False):
    file_paths = get_filepaths_by_img_num(dir_name, img_num)
    # files = []
    # dir_name_for = dir_name + "/IM-00{:02}*.dcm".format(img_num)
    # print("glob: {}".format(dir_name_for))
    # for fname in glob.glob(dir_name_for, recursive=False):
    #     files.append(pydicom.dcmread(fname))

    # print("file count: {}".format(len(files)))

    # skip files with no SliceLocation (eg scout views)
    files = [pydicom.dcmread(file_path) for file_path in file_paths]
    # slices = []
    # skipcount = 0

    slices = [slice for slice in files if hasattr(slice, "SliceLocation")]

    # for f in files:
    #     if hasattr(f, "SliceLocation"):
    #         slices.append(f)
    #     # else:
            # skipcount = skipcount + 1

    # print("skipped, no SliceLocation: {}".format(skipcount))

    # ensure they are in the correct order
    slices = sorted(slices, key=lambda sl: sl.SliceLocation)

    # assuming all slices have the same pixel aspects
    pixel_spacing = slices[0].PixelSpacing
    slice_thickness = slices[0].SliceThickness

    voxel_size = np.ndarray(shape=3, buffer=np.array([pixel_spacing[0], pixel_spacing[1], slice_thickness]), dtype=float) 

    img_shape = [*slices[0].pixel_array.shape, len(slices)]

    img3d = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img3d[:, :, i] = s.pixel_array

    img3d_scaled = scipy.ndimage.zoom(img3d, voxel_size)

    # # plot 3 orthogonal slices
    # if plot:
    #     plot_arr(img3d, img_shape, img_num, ps,ss)
        

    return img3d_scaled #, #TODO create config with voxel_size