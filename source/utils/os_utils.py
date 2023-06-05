import os
import json

def write_config_file(folder_path:str, file_prefix:str, content:dict):
    for k, v in content.items():
        try:
            json.dumps({k:v})
        except TypeError:
            content[k] = str(v)
    with open(os.path.join(folder_path, f"{file_prefix}_config.json"), "w") as f:
        f.write(json.dumps(content) )
