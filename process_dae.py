import numpy as np
import ctypes as ct
import collada
import os
import sys, getopt
from collada import Collada
from collada.camera import PerspectiveCamera, OrthographicCamera
from collada.common import DaeBrokenRefError
from collada.light import AmbientLight, DirectionalLight, PointLight, SpotLight
from collada.material import Map
from collada.polylist import Polylist, BoundPolylist
from collada.primitive import BoundPrimitive
from collada.scene import Scene, Node, NodeNode, GeometryNode
from collada.triangleset import TriangleSet, BoundTriangleSet

mean = 0
scale = 0
scale_matrix = None
inv_scale_matrix = None
input_f = None
output_f = None
dll=np.ctypeslib.load_library('clean_dae','.')

def transform_vector(x, mat, sign):
    M = np.asmatrix(mat).transpose()
    return None if x is None else np.asarray((x * M[:3, :3]) + (mat[:3, 3] * sign)).astype(np.float64)

def dfs_get(x, traingles, matrix, dep):
    if hasattr(x, 'matrix'):
       matrix = np.dot(matrix, x.matrix)
    if not hasattr(x, 'children'):
        if isinstance (x, collada.scene.GeometryNode):
            for y in x.geometry.primitives:
                if isinstance(y, collada.lineset.LineSet):
                    continue
                if not isinstance(y, collada.triangleset.TriangleSet):
                   print(type(y))
                   sys.exit(0)
                traingles += transform_vector(y.vertex, matrix, 1)[y.vertex_index].tolist()
                if y.normal_index is None or y.normal is None:
                    print('No normal Error')
                    print(type(y))
                    print(dir(y))
                    print(y.vertex_index.shape)
                    print(y.vertex.shape)
                    print(y.normal_index is None)
                    print(y.normal is None)
                    sys.exit(0)
        return
    for y in x.children:
        dfs_get(y, traingles, matrix, dep + 1)

def get_scale_mean(alltriangles):
    alltriangles = np.array(alltriangles).astype(np.float64)
    mean = (alltriangles.max(axis=(0, 1)) + alltriangles.min(axis=(0, 1))) * 0.5
    scale = np.max(alltriangles.max(axis=(0, 1)) - alltriangles.min(axis=(0, 1))) / 1.5
    return mean, scale

def importDAE(filepath):
    fin = collada.Collada(filepath)
    alltriangles, allnormals = [], []
    matrix = np.identity(4, dtype=np.float64)
    dfs_get(fin.scene.nodes[0], alltriangles, matrix, 0)
    global scale_matrix, inv_scale_matrix, mean, scale

    mean, scale = get_scale_mean(alltriangles)
    scale_matrix = np.eye(4)
    scale_matrix[:3, 3] = -mean
    scale_matrix[:3, :4] /= scale

    inv_scale_matrix = np.eye(4)
    inv_scale_matrix[:3, 3] = mean
    inv_scale_matrix[:3, :3] *= scale

    return fin

def check_large(files):
    dae = collada.Collada(files)
    alltriangles = np.vstack(
        [i.vertex[i.vertex_index]
         for j in dae.scene.objects('geometry')
         for i in j.primitives()
         if isinstance(i, collada.triangleset.BoundTriangleSet)])
    print(files, alltriangles.shape)

def print_dae(x, dep):
    if not hasattr(x, 'children'):
        if isinstance (x, collada.scene.GeometryNode):
            print(x, dep, dir(x.geometry.primitives[0]), len(x.materials), type(x.geometry))
            sys.exit(0)
        else:
            print(x)
        return
    for y in x.children:
        print_dae(y, dep + 1)

def change_transaprent(mesh):
    creator = mesh.xmlnode.getroot().getchildren()[0].getchildren()[0].getchildren()[0].text
    for mat in mesh._effects:
        if creator in ['Google 3D Warehouse 1.0', 'SketchUp 15.3.331']:
            if mat.transparency != 1:
                transparency = mat.transparency
            elif mat.transparent != None and mat.opaque_mode == 'RGB_ZERO':
                a, b, c, d = mat.transparent
                transparency = (a + b + c) / 3.0
            else:
                transparency = 0.0
        else:  # SketchUp 7.0.1
            transparency = mat.transparency
        mat.transparency = 1 - transparency
    return mesh
    
geometry_map = {}
mat_map = {}

def get_data_of_primitives(primitives):
    len_face = np.array([len(x) for x in primitives], dtype='int32')
    new_index = np.zeros(np.sum(len_face), dtype='int32')
    vertex = (primitives[0].vertex.astype(np.float64) - mean) / scale
    vertex = np.concatenate([vertex, vertex], axis=0).astype(np.float64)
    normal = primitives[0].normal.astype(np.float64)

    vertex_index = []
    normal_index = []

    def get_diff(x, y):
        return np.array_equal(x.astype(np.float64), y.astype(np.float64))

    for p in primitives:
        vertex_index.append(p.vertex_index)
        normal_index.append(p.normal_index)
        if not get_diff(p.vertex, primitives[0].vertex):
            print('Vertex Not Equal!!!')
            sys.exit(-1)
        if not get_diff(p.normal, primitives[0].normal):
            print('Normal Not Equal!!!')
            sys.exit(-1)

    vertex_index = np.concatenate(vertex_index, axis=0).astype(np.int32)
    normal_index = np.concatenate(normal_index, axis=0).astype(np.int32)
    return len_face, new_index, vertex, vertex_index, normal, normal_index
p_sign = False

def get_geom(node, mesh, mat):
    if node.geometry.id in geometry_map:
        return geometry_map[node.geometry.id]
    primitives = []
    for idx, p in enumerate(node.geometry.primitives):
        if isinstance(p, (Polylist, BoundPolylist)):
            y = p.triangleset()
        elif isinstance(p, (TriangleSet, BoundTriangleSet)):
            y = p
        else:
            continue
        if y.vertex is None or y.vertex_index is None:
            continue
        primitives.append(y)

    if len(primitives) == 0:
        geometry_map[node.geometry.id] = None
        return None
    len_face, new_index, vertex, vertex_index, normal, normal_index = get_data_of_primitives(primitives)

    eigv = np.linalg.eig(mat)[0]
    eigv = eigv[np.where(np.abs(eigv.imag) <= 1E-7)].max().real
    mean_, scale_ = get_scale_mean(vertex[vertex_index])
    scale_ *= eigv
    vertex /= scale_
    faces = dll.process(ct.c_int(len(len_face)), ct.c_int(vertex.shape[0] // 2), len_face.ctypes.data_as(ct.c_void_p),
        new_index.ctypes.data_as(ct.c_void_p), vertex.ctypes.data_as(ct.c_void_p),
        vertex_index.ctypes.data_as(ct.c_void_p), normal.ctypes.data_as(ct.c_void_p),
        normal_index.ctypes.data_as(ct.c_void_p))
    vertex *= scale_

    dicts = node.geometry.__dict__
    soure_lists = []
    soure_maps = {}
    have_vertex = None
    for atrib in dicts['sourceById']:
        atrib_list = atrib.split('-')
        source_old = dicts['sourceById'][atrib]
        if atrib_list[-1] == 'position':
            tmp = collada.source.FloatSource(source_old.id, np.array(vertex).astype(np.float32),
                                             source_old.components)
            soure_maps[source_old.id] = tmp
            soure_lists.append(tmp)
        elif atrib_list[-1] == 'normal' or atrib_list[-1] == 'uv':
            data_old = source_old.data
            tmp = collada.source.FloatSource(source_old.id, np.array(data_old).astype(np.float32),
                                             source_old.components)
            soure_maps[source_old.id] = tmp
            soure_lists.append(tmp)
        elif atrib_list[-1] == 'vertices' or atrib_list[-1] == 'vertex':
            pass
        else:
            print('Error!!', atrib_list[-1])
            sys.exit(-1)


    geom = collada.geometry.Geometry(mesh, node.geometry.id, node.geometry.name, soure_maps, double_sided=node.geometry.double_sided)
    begin_iii = 0
    for idx, y in enumerate(primitives):
        sources = y.sources
        input_list = collada.source.InputList()
        maxm = {}
        for strs in ('VERTEX', 'NORMAL', 'TEXCOORD', 'TEXBINORMAL', 'TEXTANGENT', 'COLOR', 'TANGENT', 'BINORMAL'):
            if len(sources[strs]) > 0:
                if len(sources[strs]) > 1:
                    print(len(sources[strs]), strs)
                    sys.exit(-1)
                else:
                    maxm[sources[strs][0][0]] = strs
                    input_list.addInput(sources[strs][0][0], sources[strs][0][1], sources[strs][0][2], sources[strs][0][3])
        indices = []
        begin_idx = 0 if idx == 0 else len_face[idx - 1]
        end_idx = faces if idx == len(primitives) - 1 else len_face[idx]
        idxs = new_index[begin_idx : end_idx]
        for i in range(len(maxm)):
            if i not in maxm:
                print ('Error --- 1!!')
                sys.exit(-1)
            else:
                if maxm[i] == 'VERTEX':
                    indices.append(vertex_index[idxs])
                elif maxm[i] == 'NORMAL':
                    indices.append(normal_index[idxs])
                elif maxm[i] == 'TEXCOORD':

                    if len(y.texcoord_indexset) > 1:
                        print ('Error --- 2!!')
                        sys.exit(-1)
                    else:
                        indices.append(y.texcoord_indexset[0][idxs - begin_iii])
                else:
                    print ('Error --- 3!!')
                    sys.exit(-1)
        for idxp, x in enumerate(indices):
            indices[idxp] = np.expand_dims(x.flatten(), axis=1)
        indices = np.concatenate(indices, axis=1).flatten()
        begin_iii += y.vertex_index.shape[0]
        triset = geom.createTriangleSet(indices, input_list, y.material)
        geom.primitives.append(triset)

    mat_nodes = []
    for x in node.materials:
        matnode = collada.scene.MaterialNode(x.symbol, mat_map[x.target.id], x.inputs)
        mat_nodes.append(matnode)

    mesh.geometries.append(geom)
    geom_node = collada.scene.GeometryNode(geom, mat_nodes)
    geometry_map[node.geometry.id] = geom_node
    return geom_node

mat_node_id = 0

def build_new_dae(old_dae):
    global output_f, mat_map
    mesh = collada.Collada()
    old_dae = change_transaprent(old_dae)
    mesh._effects = old_dae.effects
    mesh._materials = old_dae.materials
    mesh.zfile = old_dae.zfile
    mesh._images = old_dae.images

    for x in mesh.materials:
        mat_map[x.id] = x

    def dfs_copy(node, begin_idx, mesh, hash_node_id, mat_map, mat, dep):
        global mat_node_id
        if not hasattr(node, 'children'):
            if isinstance(node, collada.scene.GeometryNode):
                assert not hasattr(node, 'matrix')
                total_triangles = 0
                geomnode = get_geom(node, mesh, mat)
                return geomnode
            else:
                print('Not scene.GeometryNode', type(node))
                return None
        mat = np.dot(mat, node.matrix)
        matrix = np.identity(4)
        if not hasattr(node, 'matrix'):
            print(node, 'Wrong! No matrix!!')
            sys.exit(-1)
        else :
            matrix = node.matrix

        new_childeren = []
        for y in node.children:
            new_child = dfs_copy(y, begin_idx, mesh, hash_node_id, mat_map, mat, dep + 1)
            if not (new_child is None):
                new_childeren.append(new_child)
        if len(new_childeren) == 0:
            return None

        if node.id is None:
            print('No ID!')
            here_id = 'ID1'
        else :
            here_id = node.id
        if isinstance(node, collada.scene.NodeNode):
            transforms = node.node.transforms
        else:
            transforms = node.transforms
        if len(transforms) > 0:
            if len(transforms) == 1 and not isinstance(transforms[0], collada.scene.MatrixTransform):
                print(transforms[0])
            matrix = np.matmul(np.matmul(scale_matrix, matrix), inv_scale_matrix).flatten()
            transforms = [collada.scene.MatrixTransform(matrix)]
        else:
            transforms = []
        if isinstance(node, collada.scene.Node):
            new_node = collada.scene.Node(id=here_id, children=new_childeren, transforms=transforms)
        elif isinstance(node, collada.scene.NodeNode):
            tmp_node = collada.scene.Node(id=here_id, children=new_childeren, transforms=transforms)
            new_node = collada.scene.NodeNode(node=tmp_node)
        else:
            print('Not scene.Node and scene.NodeNode', type(x))
            sys.exit(-1)
        assert not (new_node is None)
        return new_node
    new_nodes = []
    begin_idx = [0]
    hash_node_id = {}
    for node in old_dae.scene.nodes:
        tmp_node = dfs_copy(node, begin_idx, mesh, hash_node_id, mat_map, np.identity(4), 1)
        if tmp_node is None:
            continue
        new_nodes.append(tmp_node)
    new_scene = collada.scene.Scene(id=old_dae.scene.id, nodes=new_nodes)
    mesh.scenes.append(new_scene)
    mesh.scene = new_scene
    print('Save new files')
    mesh.assetInfo.upaxis = 'Z_UP'
    mesh.save()

    def check_xml(xmlnode, dep):
        if dep <=3:
            print(xmlnode, dep)
        for x in xmlnode._children:
            check_xml(x, dep + 1)
    mesh.write(output_f)
    return mesh

def main(argv):
    opts, args = getopt.getopt(argv, "hi:o:", ["input=", "output="])
    global input_f, output_f
    for opt, arg in opts:
        if opt == '-h':
            print('python clean_dae -i <input_files> -o <output_files>')
            sys.exit()
        elif opt in ("-i", "--input"):
            input_f = arg
        elif opt in ("-o", "--output"):
            output_f = arg
    assert not ((input_f is None) or (output_f is None))
    filepath = input_f
    old_dae = importDAE(filepath)
    import time
    t0 = time.time()
    new_dae = build_new_dae(old_dae)
    t1 = time.time()
    print(t1 - t0)

if __name__ == "__main__":
    print('Begin to Process!')
    main(sys.argv[1:])

