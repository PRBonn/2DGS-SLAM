"""Shim: `Camera` is defined under `gaussian_splatting`; keep `from utils.camera_utils` stable."""
from gaussian_splatting.utils.camera_utils import Camera

__all__ = ["Camera"]
