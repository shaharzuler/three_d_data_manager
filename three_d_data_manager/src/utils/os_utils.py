import os
import json

def write_config_file(folder_path:str, file_prefix:str, content:dict):
    for k, v in content.items():
        try:
            json.dumps({k:v})
        except TypeError:
            content[k] = str(v)
    with open(os.path.join(folder_path, f"{file_prefix}_config.json"), "w") as f:
        f.write(json.dumps(content, indent=4) )

def read_filepaths(dataset_root_path):
    possible_path = os.path.join(dataset_root_path, "file_paths.json")
    if os.path.isfile(possible_path):
        with open(possible_path, 'r') as f:
            j_dict = json.load(f)
            return j_dict
    else:
        return None
