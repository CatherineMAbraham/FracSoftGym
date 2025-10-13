import os
import sys
from object2urdf import ObjectUrdfBuilder
import pybullet

object_folder = "241202/leg"

builder = ObjectUrdfBuilder(object_folder, urdf_prototype='_prototype.urdf')
builder.build_urdf(filename="241202/leg/leg.obj", force_overwrite=True, decompose_concave=True, force_decompose=False, center = 'mass')


