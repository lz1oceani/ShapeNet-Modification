import os
import math
from tempfile import NamedTemporaryFile
from contextlib import contextmanager
import numpy as np

import bpy
from bpy.ops import BPyOpsSubModOp
from bpy_extras.image_utils import load_image
from mathutils import Matrix, Vector

from collada import Collada
from collada.camera import PerspectiveCamera, OrthographicCamera
from collada.common import DaeBrokenRefError
from collada.light import AmbientLight, DirectionalLight, PointLight, SpotLight
from collada.material import Map
from collada.polylist import Polylist, BoundPolylist
from collada.primitive import BoundPrimitive
from collada.scene import Scene, Node, NodeNode, GeometryNode
from collada.triangleset import TriangleSet, BoundTriangleSet


__all__ = ['load']

VENDOR_SPECIFIC = []
COLLADA_NS      = 'http://www.collada.org/2005/11/COLLADASchema'
DAE_NS          = {'dae': COLLADA_NS}
TRANSPARENCY_RAY_DEPTH = 8
MAX_NAME_LENGTH        = 27

def unique_vertexset(vertex):
        vertex_new_num = 0
        vertex_new_set = {}
        vertex_new = []
        vertex_map = []

        for i in range(vertex.shape[0]):
            v = np.int32(vertex[i] * 10000)
            v = (v[0], v[1], v[2])
            if not v in vertex_new_set:
                vertex_new_set[v] = vertex_new_num
                vertex_map.append(vertex_new_num)
                vertex_new.append(vertex[i])
                vertex_new_num += 1
            else:
                vertex_map.append(vertex_new_set[v])

        print(len(vertex_new), vertex.shape)
        return np.array(vertex_new), vertex_map

def findtwoCircle(v, adj, triset, bgeom, normal, triset_normal):
    if len(adj) == 2:
        nlist = [[], []]
        for i in range(2):
            nlist[i].append(adj[i])

        return nlist

    edge = {}
    in_du = {}
    out_du = {}

    def add(x, y, id):
        if not x in edge:
            edge[x] = []

        edge[x].append((y, id))
        
        if not y in in_du:
            in_du[y] = 0
        
        if not x in out_du:
            out_du[x] = 0

        in_du[y] += 1
        out_du[x] += 1

    for tri_index in adj:
        a, b, c = triset[tri_index]
        if v == a:
            add(b, c, tri_index)
        elif v == b:
            add(c, a, tri_index)
        else:
            add(a, b, tri_index)
    
    mark = {}
    T = 0

    A = []
    B = []

    #print("edge", edge, in_du, out_du)

    while True:
        T += 1
        start = 0
        minvalue = 100000

        for k in edge.keys():
            in_degree_k = 0 if not k in in_du else in_du[k]
            out_degree_k = 0 if not k in out_du else out_du[k]

            if in_degree_k < minvalue and out_degree_k != 0:
                minvalue = in_degree_k
                start = k

        if minvalue == 100000:
            break
        
        incircle = {}
        line = []
        x = start
        ls0 = []
        ls1 = []
        
        last_normal = None

        #print("start =", x)
        while True:
            incircle[x] = 1
            line.append(x)
            #print(x)
            if not x in edge:
                break

            flag = False
            for y, tri_index in edge[x]:
                #print(x, y, tri_index, mark)

                if (not y in incircle) and (not tri_index in mark):
                    current_normal = None

                    for j in range(3):
                        if triset[tri_index][j] == v:
                            current_normal = normal[triset_normal[tri_index][j]]
                            current_normal = current_normal / np.linalg.norm(current_normal)
                            break

                    if last_normal is None:
                        last_normal = current_normal

                    if np.dot(last_normal, current_normal) < -0.2:
                        continue
                    
                    mark[tri_index] = T
                    ls0.append(tri_index)
                    in_du[y] -= 1
                    out_du[x] -= 1
                    x = y
                    flag = True
                    break

            if not flag:
                break
        
        #(line)
        iscircle = False
        if x in edge and len(line) > 2:
            for y, tri_index in edge[x]:
                if y == start  and not tri_index in mark:
                    mark[tri_index] = T
                    ls0.append(tri_index)
                    in_du[y] -= 1
                    out_du[x] -= 1
                    iscircle = True
                    break
        
        if iscircle:
            line.append(start)
        
        line.reverse()
        
        onlyone = False
        for i in range(len(line) - 1):
            x = line[i]
            if not x in edge:
                onlyone = True
                break
                
            flag = False
            for y, tri_index in edge[x]:
                if y == line[i + 1] and not tri_index in mark:
                    mark[tri_index] = T
                    ls1.append(tri_index)
                    in_du[y] -= 1
                    out_du[x] -= 1

                    flag = True
                    break
                
            if not flag:
                onlyone = True
                break
            
        if onlyone:
            for tri_index in adj:
                if not tri_index in mark:
                    ls1.append(tri_index)
            
            return [ls0, ls1]
        else:
            A.append(ls0)
            B.append(ls1)
        
    def calcnormal(ls):
        sumn = np.zeros((3))
        for tri_index in ls:
            for i in range(3):
                if triset[tri_index][i] == v:
                    sumn += normal[triset_normal[tri_index][i]]
                    break
        
        if np.linalg.norm(sumn) < 1e-5:
            return sumn
        else:
            return sumn / np.linalg.norm(sumn)

    nlist = [A[0], B[0]]
    check_normal = [calcnormal(A[0]), calcnormal(B[0])]

    for i in range(1, len(A)):
        aa = A[i]
        bb = B[i]

        aan = calcnormal(aa)
        bbn = calcnormal(bb)

        if np.dot(aan, check_normal[0]) > 0 and np.dot(bbn, check_normal[1]) > 0:
            nlist[0] += aa
            nlist[1] += bb
        elif np.dot(aan, check_normal[1]) > 0 and np.dot(bbn, check_normal[0]) > 0:
            nlist[0] += bb
            nlist[1] += aa
    '''
    if len(nlist[0]) != 0 and len(nlist[1]) != 0 and len(nlist[0]) != len(nlist[1]):
        print(bgeom.original.id)
        if 'mesh7' in bgeom.original.id:
            print(edge)
    '''
    
    return nlist
    


def load(op, ctx, filepath=None, **kwargs):
    c = Collada(filepath, ignore=[DaeBrokenRefError])
    impclass = get_import(c)
    imp = impclass(ctx, c, os.path.dirname(filepath), **kwargs)

    tf = kwargs['transformation']
    
    allvertices=np.vstack([i.vertex for j in c.scene.objects('geometry') for i in j.primitives() if isinstance(i,BoundTriangleSet)])
    scale=np.max(allvertices.max(axis=0)-allvertices.min(axis=0)) / 2
    mean=(allvertices.min(axis=0)+allvertices.max(axis=0))*0.5
    
    
    for i, obj in enumerate(c.scene.objects('geometry')):
        b_geoms = imp.geometry(obj, mean, scale)

    return {'FINISHED'}

@contextmanager
def prevented_updates(ctx):
    """ Stop Blender from funning scene update for each change. Update it
        just once the import is finished. """
    scene_update = BPyOpsSubModOp._scene_update
    setattr(BPyOpsSubModOp, '_scene_update', lambda ctx: None)
    yield
    setattr(BPyOpsSubModOp, '_scene_update', scene_update)
    BPyOpsSubModOp._scene_update(ctx)

def get_import(collada):
    for i in VENDOR_SPECIFIC:
        if i.match(collada):
            return i
    return ColladaImport


class ColladaImport(object):
    """ Standard COLLADA importer. """
    def __init__(self, ctx, collada, basedir, **kwargs):
        self._ctx = ctx
        self._collada = collada
        self._kwargs = kwargs
        self._images = {}
        self._namecount = 0
        self._names = {}
        
    def camera(self, bcam):
        bpy.ops.object.add(type='CAMERA')
        b_obj = self._ctx.object
        b_obj.name = self.name(bcam.original, id(bcam))
        b_obj.matrix_world = Matrix(bcam.matrix)
        b_cam = b_obj.data
        if isinstance(bcam.original, PerspectiveCamera):
            b_cam.type = 'PERSP'
            prop = b_cam.bl_rna.properties.get('lens_unit')
            if 'DEGREES' in prop.enum_items:
                b_cam.lens_unit = 'DEGREES'
            elif 'FOV' in prop.enum_items:
                b_cam.lens_unit = 'FOV'
            else:
                b_cam.lens_unit = prop.default
            b_cam.angle = math.radians(max(
                    bcam.xfov or bcam.yfov,
                    bcam.yfov or bcam.xfov))
        elif isinstance(bcam.original, OrthographicCamera):
            b_cam.type = 'ORTHO'
            b_cam.ortho_scale = max(
                    bcam.xmag or bcam.ymag,
                    bcam.ymag or bcam.xmag)
        if bcam.znear:
            b_cam.clip_start = bcam.znear
        if bcam.zfar:
            b_cam.clip_end = bcam.zfar


    def geometry(self, bgeom, mean, scale):
        #if not "mesh11" in bgeom.original.id:
        #    return
        b_materials = {}
        for sym, matnode in bgeom.materialnodebysymbol.items():
            mat = matnode.target
            b_matname = self.name(mat)
            if b_matname not in bpy.data.materials:
                b_matname = self.material(mat, b_matname)
            b_materials[sym] = bpy.data.materials[b_matname]

        vadj = {}
        vertex = None
        normal = None
        triset = []
        triset_normal = []
        count = 0

        for i, obj in enumerate(bgeom.primitives()):
            if not isinstance(obj, BoundTriangleSet):
                continue
            
            if vertex is None:
                vertex = obj.vertex
            if normal is None:
                normal = obj.normal

        if vertex is None:
            return

        vertex = (vertex - mean) / scale
        #vertex_new, vertex_map = unique_vertexset(vertex)
        vertex_map = range(vertex.shape[0])
        vertex_new = vertex
        
        smalltri = []
        for i, obj in enumerate(bgeom.primitives()):
            if not isinstance(obj, BoundTriangleSet):
                continue
            for j in range(obj.vertex_index.shape[0]):
                a = vertex_map[obj.vertex_index[j][0]]
                b = vertex_map[obj.vertex_index[j][1]]
                c = vertex_map[obj.vertex_index[j][2]]
                an = obj.normal_index[j][0]
                bn = obj.normal_index[j][1]
                cn = obj.normal_index[j][2]

                triangle = (a, b, c)
                triset.append(triangle)
                triset_normal.append((an, bn, cn))
                n = np.cross(vertex[a] - vertex[b], vertex[a] - vertex[c])
                
                if a == b or a == c or b == c or np.linalg.norm(n) < 1e-8:
                    smalltri.append(True)
                else:
                    smalltri.append(False)
                
                    for x in [a, b, c]:
                        if not x in vadj:
                            vadj[x] = []
                        
                        vadj[x].append(count)

                count += 1

            

        smalltri = np.array(smalltri)
        vertex = vertex_new
        vertex_map = range(vertex.shape[0])
        V = np.concatenate([vertex, vertex], axis = 0)
        vertex_num = vertex.shape[0]
        tri_num = len(triset)
        col = np.zeros((tri_num, 3), dtype = 'int32')

        for v in vadj:
            adj = vadj[v]
            nlist = findtwoCircle(v, adj, triset, bgeom, normal, triset_normal)
            
            '''
            if len(nlist[0]) != 0 and len(nlist[1]) != 0 and len(nlist[0]) != len(nlist[1]):
                print(nlist[0])
                print(nlist[1])
            '''
            movedir = np.zeros((2, 3))
            for i in [0, 1]:
                for tri_index in nlist[i]:
                    a, b, c = triset[tri_index]
                    n = np.cross(vertex[a] - vertex[b], vertex[a] - vertex[c])         
                    if np.linalg.norm(n) < 1e-8:
                        n = np.zeros((3))
                    else:
                        n /= np.linalg.norm(n)
                    
                    movedir[i] += n

                    k = 0

                    if b == v:
                        k = 1
                    if c == v:
                        k = 2

                    col[tri_index, k] = i
                if np.linalg.norm(movedir[i]) < 1e-6:
                    movedir[i] = np.zeros((3))
                else:
                    movedir[i] /= np.linalg.norm(movedir[i])
                
                mvdir = 0
                for tri_index in nlist[i]:
                    for j in range(3):
                        if triset[tri_index][j] == v:
                            mvdir += normal[triset_normal[tri_index][j]]
                            break
                
                movedir[i] = mvdir
                if np.linalg.norm(movedir[i]) < 1e-6:
                    movedir[i] = np.zeros((3))
                else:
                    movedir[i] /= np.linalg.norm(movedir[i])
                
                
            eps = 1e-4
            #print(movedir)

            for i in [0, 1]:
                 V[v + i * vertex_num] += movedir[i] * eps
        #print (V)

        count = 0
        
        
        for i, obj in enumerate(bgeom.primitives()):    
            if isinstance(obj, BoundPrimitive):
                b_mat_key = obj.original.material
            else:
                b_mat_key = obj.material
            b_mat = b_materials.get(b_mat_key, None)
            b_meshname = self.name(bgeom.original, i)

            if not isinstance(obj, BoundTriangleSet):
                continue
            
            if obj.vertex is None or obj.vertex_index is None:
                continue

            b_mesh = bpy.data.meshes.new(b_meshname)
            b_mesh.vertices.add(V.shape[0])
            

            for j in range(V.shape[0]):
                b_mesh.vertices[j].co = V[j]
            
            VI = []
            valid = []
            valid_index = []
            valid_uv_index = []

            for j in range(obj.vertex_index.shape[0]):
                if not smalltri[count]:
                    VI.append([vertex_map[obj.vertex_index[j, k]] + col[count, k] * vertex_num for k in range(3)])
                    valid_index.append(count)
                    valid_uv_index.append(j)
                
                valid.append(smalltri[count])
                count += 1
            print(np.array(VI).shape)
            b_mesh.tessfaces.add(len(VI))
            eekadoodle_faces = [v
                    for f in VI
                    for v in _eekadoodle_face(*f)]

            b_mesh.tessfaces.foreach_set(
                'vertices_raw', eekadoodle_faces)

            has_normal = (obj.normal_index is not None)
            has_uv = (len(obj.texcoord_indexset) > 0)

            if has_normal:
                # TODO import normals
                for i, f in enumerate(b_mesh.tessfaces):
                    f.use_smooth = not _is_flat_face(
                            normal[list(triset_normal[valid_index[i]])])
            
            if has_uv:
                

                for j in range(len(obj.texcoord_indexset)):
                    #print("tessfaces", len(b_mesh.tessfaces))
                    #print(len(obj.texcoord_indexset[j]), len(obj.texcoordset[j]))
                    self.texcoord_layer(
                            obj,
                            valid_uv_index,
                            obj.texcoordset[j],
                            obj.texcoord_indexset[j],
                            b_mesh,
                            b_mat)
            

            b_mesh.update()

            b_obj = bpy.data.objects.new(b_meshname, b_mesh)
            b_obj.data = b_mesh

            self._ctx.scene.objects.link(b_obj)
            self._ctx.scene.objects.active = b_obj

            if len(b_obj.material_slots) == 0:
                bpy.ops.object.material_slot_add()
            b_obj.material_slots[0].link = 'OBJECT'
            b_obj.material_slots[0].material = b_mat
            b_obj.active_material = b_mat
            
            if self._transform('APPLY'):
                # TODO import normals
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.normals_make_consistent()
                bpy.ops.object.mode_set(mode='OBJECT')


    def texcoord_layer(self, triset, valid_index, texcoord, index, b_mesh, b_mat):
        b_mesh.uv_textures.new()
        for i, f in enumerate(b_mesh.tessfaces):
            t1, t2, t3 = index[valid_index[i]]
            tface = b_mesh.tessface_uv_textures[-1].data[i]
            # eekadoodle
            if triset.vertex_index[valid_index[i]][2] == 0:
                t1, t2, t3 = t3, t1, t2
            tface.uv1 = texcoord[t1]
            tface.uv2 = texcoord[t2]
            tface.uv3 = texcoord[t3]

    def light(self, light, i):
        if isinstance(light.original, AmbientLight):
            return
        b_name = self.name(light.original, i)
        if b_name not in bpy.data.lamps:
            if isinstance(light.original, DirectionalLight):
                b_lamp = bpy.data.lamps.new(b_name, type='SUN')
            elif isinstance(light.original, PointLight):
                b_lamp = bpy.data.lamps.new(b_name, type='POINT')
                b_obj = bpy.data.objects.new(b_name, b_lamp)
                self._ctx.scene.objects.link(b_obj)
                b_obj.matrix_world = Matrix.Translation(light.position)
            elif isinstance(light.original, SpotLight):
                b_lamp = bpy.data.lamps.new(b_name, type='SPOT')

    def material(self, mat, b_name):
        effect = mat.effect
        b_mat = bpy.data.materials.new(b_name)
        b_name = b_mat.name
        b_mat.diffuse_shader = 'LAMBERT'
        getattr(self, 'rendering_' + \
                effect.shadingtype)(mat, b_mat)
        bpy.data.materials[b_name].use_transparent_shadows = \
                self._kwargs.get('transparent_shadows', False)
        if effect.emission:
            b_mat.emit = sum(effect.emission[:3]) / 3.0
        self.rendering_transparency(effect, b_mat)
        self.rendering_reflectivity(effect, b_mat)
        return b_name

    def node(self, node, parent):
        if isinstance(node, (Node, NodeNode)):
            b_obj = bpy.data.objects.new(self.name(node), None)
            b_obj.matrix_world = Matrix(node.matrix)
            self._ctx.scene.objects.link(b_obj)
            if parent:
                b_obj.parent = parent
            parent = b_obj
        elif isinstance(node, GeometryNode):
            for bgeom in node.objects('geometry'):
                b_geoms = self.geometry(bgeom)
                for b_obj in b_geoms:
                    b_obj.parent = parent
        return parent

    def rendering_blinn(self, mat, b_mat):
        effect = mat.effect
        b_mat.specular_shader = 'BLINN'
        self.rendering_diffuse(effect.diffuse, b_mat)
        self.rendering_specular(effect, b_mat)

    def rendering_constant(self, mat, b_mat):
        b_mat.use_shadeless = True

    def rendering_lambert(self, mat, b_mat):
        effect = mat.effect
        self.rendering_diffuse(effect.diffuse, b_mat)
        b_mat.specular_intensity = 0.0

    def rendering_phong(self, mat, b_mat):
        effect = mat.effect
        b_mat.specular_shader = 'PHONG'
        self.rendering_diffuse(effect.diffuse, b_mat)
        self.rendering_specular(effect, b_mat)

    def rendering_diffuse(self, diffuse, b_mat):
        b_mat.diffuse_intensity = 1.0
        diff = self.color_or_texture(diffuse, b_mat)
        if isinstance(diff, tuple):
            b_mat.diffuse_color = diff
        else:
            diff.use_map_color_diffuse = True

    def rendering_specular(self, effect, b_mat):
        if effect.specular:
            b_mat.specular_intensity = 1.0
            b_mat.specular_color = effect.specular[:3]
        if effect.shininess:
            b_mat.specular_hardness = effect.shininess

    def rendering_reflectivity(self, effect, b_mat):
        if effect.reflectivity and effect.reflectivity > 0:
            b_mat.raytrace_mirror.use = True
            b_mat.raytrace_mirror.reflect_factor = effect.reflectivity
            if effect.reflective:
                refi = self.color_or_texture(effect.reflective, b_mat)
                if isinstance(refi, tuple):
                    b_mat.mirror_color = refi
                else:
                    # TODO use_map_mirror or use_map_raymir ?
                    pass

    def rendering_transparency(self, effect, b_mat):
        creator = self._collada.xmlnode.getroot().getchildren()[0].getchildren()[0].getchildren()[0].text
        if not effect.transparency:
            return
        if isinstance(effect.transparency, float):
            if creator in ['Google SketchUp 7.0.1']:
                if effect.transparency > 0.0:
                    b_mat.use_transparency = True
                    b_mat.alpha = 1.0 - effect.transparency
            else:
                if effect.transparency < 1.0:
                    b_mat.use_transparency = True
                    b_mat.alpha = effect.transparency
        
        if self._kwargs.get('raytrace_transparency', False):
            b_mat.transparency_method = 'RAYTRACE'
            b_mat.raytrace_transparency.ior = 1.0
            b_mat.raytrace_transparency.depth = TRANSPARENCY_RAY_DEPTH
        if isinstance(effect.index_of_refraction, float):
            b_mat.transparency_method = 'RAYTRACE'
            b_mat.raytrace_transparency.ior = effect.index_of_refraction
            b_mat.raytrace_transparency.depth = TRANSPARENCY_RAY_DEPTH

    def color_or_texture(self, color_or_texture, b_mat):
        if isinstance(color_or_texture, Map):
            image = color_or_texture.sampler.surface.image
            mtex = self.try_texture(image, b_mat)
            return mtex or (1., 0., 0.)
        elif isinstance(color_or_texture, tuple):
            return color_or_texture[:3]

    def try_texture(self, c_image, b_mat):
        mtex = None
        with self._tmpwrite(c_image.path, c_image.data) as tmp:
            image = load_image(tmp)
            if image is not None:
                image.pack(True)
                texture = bpy.data.textures.new(name='Kd', type='IMAGE')
                texture.image = image
                mtex = b_mat.texture_slots.add()
                mtex.texture_coords = 'UV'
                mtex.texture = texture
                self._images[b_mat.name] = image
        return mtex

    def name(self, obj, index=0):
        """ Trying to get efficient and human readable name, workarounds
        Blender's object name limitations.
        """
        if hasattr(obj, 'id'):
            uid = obj.id.replace('material', 'm')
        else:
            self._namecount += 1
            uid = 'Untitled.' + str(self._namecount)
        base = '%s-%d' % (uid, index)
        if base not in self._names:
            self._namecount += 1
            self._names[base] = '%s-%.4d' % (base[:MAX_NAME_LENGTH], self._namecount)
        return self._names[base]

    @contextmanager
    def _tmpwrite(self, relpath, data):
        with NamedTemporaryFile(suffix='.' + relpath.split('.')[-1]) as out:
            out.write(data)
            out.flush()
            yield out.name

    def _transform(self, t):
        return self._kwargs['transformation'] == t


class SketchUpImport(ColladaImport):
    """ SketchUp specific COLLADA import. """

    def rendering_diffuse(self, diffuse, b_mat):
        """ Imports PNG textures with alpha channel. """
        ColladaImport.rendering_diffuse(self, diffuse, b_mat)
        if isinstance(diffuse, Map):
            if b_mat.name in self._images:
                image = self._images[b_mat.name]
                if image.depth == 32:
                    diffslot = None
                    for ts in b_mat.texture_slots:
                        if ts and ts.use_map_color_diffuse:
                            diffslot = ts
                            break
                    if not diffslot:
                        return

                    image.use_alpha = True
                    diffslot.use_map_alpha = True
                    tex = diffslot.texture
                    tex.use_mipmap = True
                    tex.use_interpolation = True
                    tex.use_alpha = True
                    
                    #b_mat.use_transparency = True
                    #b_mat.alpha = 1.0
                    if self._kwargs.get('raytrace_transparency', False):
                        b_mat.transparency_method = 'RAYTRACE'
                        b_mat.raytrace_transparency.ior = 1.0
                        b_mat.raytrace_transparency.depth = TRANSPARENCY_RAY_DEPTH

    def rendering_phong(self, mat, b_mat):
        super().rendering_lambert(mat, b_mat)

    def rendering_reflectivity(self, effect, b_mat):
        """ There are no reflectivity controls in SketchUp """
        if not self.__class__.test2(effect.xmlnode):
            ColladaImport.rendering_reflectivity(self, effect, b_mat)

    @classmethod
    def match(cls, collada):
        xml = collada.xmlnode
        return cls.test1(xml) or cls.test2(xml)

    @classmethod
    def test1(cls, xml):
        src = [xml.find('.//dae:instance_visual_scene',
                    namespaces=DAE_NS).get('url')]
        at = xml.find('.//dae:authoring_tool', namespaces=DAE_NS)
        if at is not None:
            src.append(at.text)
        return any(['SketchUp' in s for s in src if s])

    @classmethod
    def test2(cls, xml):
        et = xml.findall('.//dae:extra/dae:technique',
                namespaces=DAE_NS)
        return len(et) and any([
            t.get('profile') == 'GOOGLEEARTH'
            for t in et])

VENDOR_SPECIFIC.append(SketchUpImport)


def _is_flat_face(normal):
    a = Vector(normal[0])
    for n in normal[1:]:
        dp = a.dot(Vector(n))
        if dp < 0.9999 or dp > 1.0001:
            return False
    return True


def _eekadoodle_face(v1, v2, v3):
    return v3 == 0 and (v3, v1, v2, 0) or (v1, v2, v3, 0)


def _children(node):
    if isinstance(node, Scene):
        return node.nodes
    elif isinstance(node, Node):
        return node.children
    elif isinstance(node, NodeNode):
        return node.node.children
    else:
        return []


def _dfs(node, cb, parent=None):
    """ Depth first search taking a callback function.
    Its return value will be passed recursively as a parent argument.

    :param node: COLLADA node
    :param callable cb:
     """
    parent = cb(node, parent)
    for child in _children(node):
        _dfs(child, cb, parent)
