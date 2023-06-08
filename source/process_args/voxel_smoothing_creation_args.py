from dataclasses import dataclass

@dataclass
class VoxelSmoothingCreationArgs:
    opening_footprint_radius: int
    fill_holes_Area_threshold: int
    closing_to_opening_ratio: float
