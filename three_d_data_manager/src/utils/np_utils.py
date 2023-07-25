import os

import numpy as np

def save_arr(output_folder_path:str, filename:str, arr:np.ndarray) -> str:
    os.makedirs(output_folder_path, exist_ok=True)
    output_file_path = os.path.join(output_folder_path, f"{filename}.npy")
    np.save(output_file_path, arr)
    return output_file_path