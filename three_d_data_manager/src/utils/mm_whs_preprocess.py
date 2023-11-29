import nibabel as nib
import numpy as np
from scipy import ndimage

def get_img_label_pair(img_path, label_path):
    img_obj = nib.load(img_path)
    label_obj = nib.load(label_path)

    zoom = np.array(img_obj.header.get_zooms())

    img_zoomed = ndimage.zoom(img_obj.get_fdata(), zoom=zoom)
    label_arr = label_obj.get_fdata() 
    reduced_LV_label = np.where( (np.logical_or(label_arr==205., label_arr==500.)), True, False)
    LV_label_zoomed = ndimage.zoom(reduced_LV_label, zoom=zoom, order=0, mode="nearest") 
    reduced_shell_label = np.where( (label_arr==205.), True, False)
    shell_label_zoomed = ndimage.zoom(reduced_shell_label, zoom=zoom, order=0, mode="nearest") 
    
    print(f"img shape: {img_zoomed.shape}, label_shape: {LV_label_zoomed.shape}")

    np.save(img_path.replace(".nii.gz", ".npy"), img_zoomed)
    np.save(label_path.replace(".nii.gz", "_LV.npy"), LV_label_zoomed)
    np.save(label_path.replace(".nii.gz", "_shell.npy"), shell_label_zoomed)

if __name__ == "__main__":
    for img_num in [1001, 1003, 1014, 1016, 1020]:
        get_img_label_pair(
            img_path=f"/home/shahar/data/mm_whs/{img_num}/ct_train_{img_num}_image.nii.gz", 
            label_path=f"/home/shahar/data/mm_whs/{img_num}/ct_train_{img_num}_label.nii.gz")






    