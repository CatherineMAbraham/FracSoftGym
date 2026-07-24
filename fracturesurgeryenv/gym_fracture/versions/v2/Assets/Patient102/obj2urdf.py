import os
import sys
from object2urdf import ObjectUrdfBuilder
import trimesh
import argparse

print(hasattr(trimesh.interfaces, 'vhacd'))


object_folder = "./"
def ob2urdf(obj):
    for ob in obj:
        builder = ObjectUrdfBuilder(object_folder, urdf_prototype='_prototype.urdf')
        builder.build_urdf(filename=ob, force_overwrite=False,
                        decompose_concave=True, force_decompose=False, center='mass')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert .obj files to URDF format.')
    parser.add_argument('--obj', type= str,nargs='+', help='Path to the .obj file(s) to convert to URDF.')
    args = parser.parse_args()
    ob2urdf(args.obj)

    