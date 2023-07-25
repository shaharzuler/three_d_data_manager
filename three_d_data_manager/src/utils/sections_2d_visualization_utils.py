from dataclasses import dataclass
import os
from typing import Dict, List

from PIL import Image
import numpy as np
from PIL import Image
import cv2

from three_d_data_manager.src.file_paths import FilePaths


@dataclass
class colors:
    YELLOW_RGB: tuple = 255, 255, 1
    RED_RGB: tuple    = 255, 27,  62
    PURPLE_RGB: tuple = 255, 1,   255
    BLUE_RGB: tuple   = 0,   102, 255
    GREEN_RGB: tuple  = 1,   255, 1


def draw_2d_sections(arr:np.array, save_path:str, pad_val = 10) -> np.array:
    arr = arr.astype(float)
    min_val, max_val = arr.min(), arr.max()
    arr_sections = np.array(arr.shape)//2
    x_section = arr[ arr_sections[0], :              , :               ]
    y_section = arr[ :              , arr_sections[1], :               ]
    z_section = arr[ :              , :              , arr_sections[2] ]
    sections_image_top = np.hstack((x_section, np.ones((arr.shape[0], arr.shape[1] + pad_val)) * min_val))
    sections_image_bottom = np.hstack((y_section, np.ones((arr.shape[0], pad_val)) * min_val, z_section))
    pad_top_from_bottom =  np.ones((pad_val, arr.shape[2] + pad_val + arr.shape[0])) * min_val
    sections_image = np.vstack((sections_image_top, pad_top_from_bottom, sections_image_bottom))
    sections_image = (((sections_image - min_val) / (max_val - min_val)) * 255).astype(np.uint8)

    if save_path is not None:
        Image.fromarray(sections_image).save(save_path, quality=100) 

    return sections_image

def draw_2d_mask_on_scan(sections_image:np.array, sections_mask:np.array, color:tuple, save_path:str) -> np.array: 
    if len(sections_image.shape) == 2:
        img_rgb = cv2.cvtColor(sections_image, cv2.COLOR_GRAY2RGB) # H x W x 3
    sections_mask = sections_mask.astype(float)/255
    mask_3_channels = np.repeat(np.expand_dims(sections_mask, -1), 3, axis=2) # H x W x 3
    mask_rgb = mask_3_channels.copy()
    mask_rgb[:,:,0] *= color[0]
    mask_rgb[:,:,1] *= color[1]
    mask_rgb[:,:,2] *= color[2]
    mask_rgb = mask_rgb.astype(sections_image.dtype)

    img_w_mask = np.where(mask_3_channels != 0, mask_rgb, img_rgb)
    if save_path is not None:
        Image.fromarray(img_w_mask, mode="RGB").save(save_path, quality=100) 

    return img_w_mask

def draw_masks_and_contours(sections_image:np.array, masks_data:List[Dict[str, any((str, np.array, tuple))]], target_root_dir:str, file_paths:FilePaths, name:str) -> FilePaths:
    contours_arr = cv2.cvtColor(sections_image, cv2.COLOR_GRAY2BGR)
    for n, mask_data in enumerate(masks_data):
        contours_arr, file_paths = _draw_single_mask_and_contour(sections_image, contours_arr, n, mask_data, target_root_dir, file_paths, name)
    
    section_contours_path = os.path.join(target_root_dir, "section_contours.jpg")
    file_paths.add_path("section_contours",  name, section_contours_path)
    Image.fromarray(cv2.cvtColor(contours_arr, cv2.COLOR_BGR2RGB), mode="RGB").save(section_contours_path, quality=100) 
    
    return file_paths

def _draw_single_mask_and_contour(sections_image:np.array, contours_arr:np.array, n:int, mask_data:List[Dict[str, any((str, np.array, tuple))]], target_root_dir:str, file_paths, name:str):
    mask_sections_path = os.path.join(target_root_dir, f"n{n}_{mask_data['name']}.jpg")
    file_paths.add_path(f"n{n}_{mask_data['name']}", name, mask_sections_path)

    sections_mask = _draw_mask(sections_image, mask_data, mask_sections_path) 
    contours_arr = _draw_mask_contours(contours_arr, n, mask_data, sections_mask)

    return contours_arr, file_paths

def _draw_mask_contours(contours_arr:np.array, n:int, mask_data:dict, sections_mask:np.array) -> np.array: 
    contours, hierarchy = cv2.findContours(sections_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    color_bgr = mask_data["color"][::-1]
    cv2.drawContours(contours_arr, contours, -1, color_bgr, 1)
    cv2.putText(contours_arr, text=mask_data['name'], org=(150,(n+1)*10),fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=color_bgr, thickness=1, lineType=cv2.LINE_AA)
    return contours_arr

def _draw_mask(sections_image, mask_data, mask_sections_path):
    sections_mask = draw_2d_sections(mask_data["arr"], save_path=None)
    sections_image_w_mask = draw_2d_mask_on_scan(sections_image, sections_mask, mask_data["color"], mask_sections_path)
    return sections_mask

