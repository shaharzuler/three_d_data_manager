import pydicom
import numpy as np
import glob
import scipy


# original code from: https://github.com/gallif/_4DCTCostUnrolling with some modifications
def images_to_3d_arr(dir_name, img_num):#, plot=False):
    files = []
    dir_name_for = dir_name + "/IM-00{:02}*.dcm".format(img_num)
    print("glob: {}".format(dir_name_for))
    for fname in glob.glob(dir_name_for, recursive=False):
        files.append(pydicom.dcmread(fname))

    print("file count: {}".format(len(files)))

    # skip files with no SliceLocation (eg scout views)
    slices = []
    # skipcount = 0
    for f in files:
        if hasattr(f, "SliceLocation"):
            slices.append(f)
        # else:
            # skipcount = skipcount + 1

    # print("skipped, no SliceLocation: {}".format(skipcount))

    # ensure they are in the correct order
    slices = sorted(slices, key=lambda sl: sl.SliceLocation)

    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness

    voxel_size = np.ndarray(shape=3, buffer=np.array([ps[0], ps[1], ss]), dtype=float) 

    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d

    img3d_scaled = scipy.ndimage.zoom(img3d, voxel_size)

    # # plot 3 orthogonal slices
    # if plot:
    #     plot_arr(img3d, img_shape, img_num, ps,ss)
        

    return img3d_scaled #, #TODO create config with voxel_size