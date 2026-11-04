bl_info = {
    "name": "Armor27 Dataset Exporter",
    "author": "armor27",
    "version": (1, 0, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > 数据集",
    "description": "Generate, stop, resume, and check armor plate datasets",
    "category": "Render",
}

"""Armor27 dataset exporter Blender add-on."""

from .batch_panel import register, unregister

__version__ = ".".join(str(part) for part in bl_info["version"])

__all__ = ("bl_info", "register", "unregister")
