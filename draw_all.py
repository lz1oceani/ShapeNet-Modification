import sys, getopt
import collada
import os
import os.path
import numpy as np
import pickle

direct = None
outputdir = None
number = -1
step = 0
visited = set()
visited_file = ''
tot_cnt = 0

def getdir(name):
    if not os.path.exists(name):
        os.mkdir(name)

def dfs(dir_in, out_image, out_dae):
    files = os.listdir(dir_in)
    global step, visited, visited_file, tot_cnt
    getdir(out_image)
    getdir(out_dae)
    for file in files:
        if len(file.split('.')) == 1:
            if not os.path.isdir(dir_in + file + '/'):
                os.system('rm ' + dir_in + file)
            else:
                dfs(dir_in + file + '/', out_image + file + '/', out_dae + file + '/')
        else:
            if step >= number:
                sys.exit(0)
            dae_in = dir_in + file
            file_name = file.split('.')[0]

            image_out = out_image + file_name + '.png'
            tot_cnt += 1

            if file_name in visited:
                continue
            else:
                visited.add(file_name)

            if file.split('.')[1] == 'dae':
                dae_out = out_dae + file_name + '.dae'
                os.system('python process_dae.py -i ' + dae_in + ' -o ' + dae_out)
                os.system('blender278 -b -P render_dae.py -- -i ./tmp/models/model.dae -o ' + image_out)
            else:
                dae_out = out_dae + file_name + '.kmz'
                os.system('rm -r ./tmp/*')
                os.system('unzip ' + dae_in + ' -d ./tmp/')
                sign = 0
                sign = os.system('python process_dae.py -i ./tmp/models/model.dae -o ./tmp/models/model.dae')
                print (sign)
                if sign // 256 != 0:
                    print(sign // 256)
                    sys.exit(-1)
                os.system('rm -r ./tmp/tmp.blend')
                os.chdir('./tmp/')
                os.system('zip -r ../' + dae_out + ' ./*')
                os.chdir('../')
                os.system('blender278 -b -P render_dae.py -- -i ./tmp/models/model.dae -o ' + image_out)
            step += 1
            if step % 1 == 0:
                pickle.dump(visited, open(visited_file, "wb"))
            

def main(argv):
    print(argv)
    global direct, number, visited, visited_file
    getdir('./tmp/')

    opts, args = getopt.getopt(argv, "hn:d:", ["num=", "dir="])
    number = None
    direct = None
    for opt, arg in opts:
        if opt == '-h':
            print('python draw_all.py -n <number of files> -d <dir>')
            sys.exit()
        elif opt in ("-n", "--num"):
            number = int(arg)
        elif opt in ("-d", "--dir"):
            direct = arg
    assert direct != None
    if direct[-1] == '/':
        direct = direct[:-1]
    outputdir_tmp = direct.split('/')
    outputdir = ""
    for i, x in enumerate(outputdir_tmp):
        outputdir += x
        if i < len(outputdir_tmp) - 1:
            outputdir += '/'
    visited_file = outputdir + '_log.pkl'
    if os.path.isfile(visited_file):
        print('Has log!!')
        visited = pickle.load(open(visited_file, "rb"))
    else:
        visited = set()

    print(direct, outputdir + '_image/')
    dfs(direct + '/', outputdir + '_image_lz/', outputdir + '_dae_lz/')

if __name__ == '__main__':
    main(sys.argv[1:])


