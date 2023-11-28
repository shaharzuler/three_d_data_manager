from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import os

ts_x_img = "/home/shahar/data/cardiac_3d_data_magix/TIMESTEP/orig/voxels/xyz_arr_raw.npy"
ts_x_mask = "/home/shahar/data/cardiac_3d_data_magix/TIMESTEP/orig/voxels/xyz_voxels_mask_smooth.npy"
output_dir = "/home/shahar/data/cardiac_3d_data_magix/sections"
for ts in range(0,100,10):
    ts_img = ts_x_img.replace("TIMESTEP", str(ts))
    ts_mask = ts_x_mask.replace("TIMESTEP", str(ts))
    img = np.load(ts_img)
    mask = np.load(ts_mask)

    plt.imshow(img[:,:,114],  cmap=mpl.colormaps["bone"]) 
    plt.savefig(os.path.join(output_dir, f"gt_img_x_{ts}.jpg"))
    plt.imshow(mask[:,:,114],  cmap=mpl.colormaps["bone"])
    plt.savefig(os.path.join(output_dir, f"gt_mask_x_{ts}.jpg"))
    plt.imshow(img[:,103,:],  cmap=mpl.colormaps["bone"]) 
    plt.savefig(os.path.join(output_dir, f"gt_img_y_{ts}.jpg"))
    plt.imshow(mask[:,103,:],  cmap=mpl.colormaps["bone"])
    plt.savefig(os.path.join(output_dir, f"gt_mask_y_{ts}.jpg"))
    plt.imshow(img[103,:,:],  cmap=mpl.colormaps["bone"]) 
    plt.savefig(os.path.join(output_dir, f"gt_img_z_{ts}.jpg"))
    plt.imshow(mask[103,:,:],  cmap=mpl.colormaps["bone"])
    plt.savefig(os.path.join(output_dir, f"gt_mask_z_{ts}.jpg"))