import numpy as np
from skimage import morphology


def xyz_to_zxy(xyz_arr:np.array):
    zxy_arr = np.moveaxis(xyz_arr,-1,0)
    return zxy_arr

def zxy_to_xyz(zxy_arr:np.array):
    xyz_arr = np.moveaxis(zxy_arr,0,-1)
    return xyz_arr


def fill_holes(masks_arr:np.array, area_threshold:int) -> np.array:
    masks_arr = morphology.remove_small_holes(masks_arr, area_threshold)
    return masks_arr

def _create_footprint(r:int) -> np.array:
    footprint = np.zeros([r,r,r])
    for i in range(r):
        for j in range(r):
            for k in range(r):
                if np.sum(np.square(np.array([i,j,k])-np.array([2,2,2])))<2*(r+1):
                    footprint[i,j,k]=1
    return footprint

def voxel_smoothing(masks_arr:np.array, opening_footprint_radius:int, closing_to_opening_ratio:float) -> np.array: 
    footprint_small = _create_footprint(r=int(opening_footprint_radius*closing_to_opening_ratio))
    closed_masks_arr = morphology.closing(masks_arr, footprint_small) 
    footprint = _create_footprint(r=opening_footprint_radius)
    smooth_masks_arr = morphology.opening(closed_masks_arr, footprint)  
    return smooth_masks_arr



