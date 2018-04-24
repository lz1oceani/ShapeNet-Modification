import bpy
import sys, getopt
import collada
from collada import *
import os
import numpy as np

if __name__ == '__main__':
    bpy.ops.wm.open_mainfile(filepath='./tmp/tmp.blend')
    #bpy.context.scene.view_render.engine = 'BLENDER_EEVEE'

    bpy.ops.render.render(write_still=True)


