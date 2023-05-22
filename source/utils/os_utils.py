import os
import json

def write_config_file(folder_path:str, file_prefix:str, content:dict):
    with open(os.path.join(folder_path, f"{file_prefix}_config.json"), "w") as f:
        f.write(json.dumps(content) )
