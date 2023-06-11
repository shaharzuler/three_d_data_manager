from dataclasses import dataclass

@dataclass
class H5DatasetCreationArgs:
    orig_name: str
    override: bool = False

    