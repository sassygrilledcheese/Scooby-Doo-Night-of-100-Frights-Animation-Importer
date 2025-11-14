bl_info = {
    "name": "Scooby Animation Importer",
    "author": "Sassy Grilled Cheese",
    "version": (1, 4, 0),
    "blender": (4, 0, 0),
    "location": "File > Import > Scooby Animation (.ska, .anm)",
    "description": "Import Scooby-Doo / BFBB SKA animations and ANM animations onto the selected armature.",
    "category": "Import-Export",
}

import importlib
import bpy

# Import your main module inside the package
from . import Scooby_Animation_Importer

# Forward registration to the main file
def register():
    Scooby_Animation_Importer.register()

def unregister():
    Scooby_Animation_Importer.unregister()

