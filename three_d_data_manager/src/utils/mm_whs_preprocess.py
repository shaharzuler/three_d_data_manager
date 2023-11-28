import nibabel as nib
import numpy as np
from scipy import ndimage

def get_img_label_pair(img_path, label_path):
    img_obj = nib.load(img_path)
    label_obj = nib.load(label_path)

    zoom = np.array(img_obj.header.get_zooms())

    img_zoomed = ndimage.zoom(img_obj.get_fdata(), zoom=zoom)
    label_arr = label_obj.get_fdata() 
    reduced_label = np.where( (np.logical_or(label_arr==205., label_arr==500.)), True, False)
    label_zoomed = ndimage.zoom(reduced_label, zoom=zoom, order=0, mode="nearest") 
    print(f"img shape: {img_zoomed.shape}, label_shape: {label_zoomed.shape}")

    np.save(img_path.replace(".nii.gz", ".npy"), img_zoomed)
    np.save(label_path.replace(".nii.gz", ".npy"), label_zoomed)

for img_num in [1001, 1003, 1014, 1016, 1020]:
    get_img_label_pair(
        img_path=f"/home/shahar/data/mm_whs/{img_num}/ct_train_{img_num}_image.nii.gz", 
        label_path=f"/home/shahar/data/mm_whs/{img_num}/ct_train_{img_num}_label.nii.gz")






    