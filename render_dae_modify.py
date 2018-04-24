import bpy
import sys, getopt, os

def draw(input, output):
    bpy.ops.wm.open_mainfile(filepath='./init_278.blend')
    bpy.ops.import_scene_modify.collada(filepath=input)
    print('Begin to render ' + input + ' ' + output)
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.context.scene.render.image_settings.compression = 100
    bpy.context.scene.render.filepath = output
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080

def main(argv):
    print(argv)
    opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    inputfile = None
    outputfile = None
    for opt, arg in opts:
        if opt == '-h':
            print('blender2 -b -P dae2blend.py -- -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    assert inputfile != None
    assert outputfile != None
    print('Input file is ', inputfile)
    print('Output file is ', outputfile)
    draw(inputfile, outputfile)
    print('End of Render!')


if __name__ == '__main__':
    tmp = None
    for idx, x in enumerate(sys.argv):
        if x == '--':
            tmp = sys.argv[idx + 1: ]
    main(tmp)


