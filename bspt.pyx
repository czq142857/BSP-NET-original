import numpy as np
from math import sqrt
cimport cython

# to build this file:
# python setup.py build_ext --inplace

#input format
#convex_list -> [convex,convex,...]
#convex -> [face,face,...]
#face -> [a,b,c,d]
#ax+by+cz+d=0 is the plane
#ax+by+cz+d>0 is the inner part

#note: assume the mesh is convex and within [-1,1]^3
#face: [ [nx,ny,nz,w], v1,v2,v3,... ]
#nx,ny,nz is normal = a,b,c
#w = dot(normal,v1) = -d distance from origin to the plane
cdef float border_limit = 1.0
cdef float epsilon1 = 1e-5
cdef float epsilon2 = 1e-5

#convert a parametric face to a polygon
@cython.boundscheck(False)
@cython.wraparound(False)
def get_polygon_from_params(float[:, :, ::1] faces, float[:, ::1] face_params, Py_ssize_t idx, float a, float b, float c, float d):
    if a*a+b*b+c*c<epsilon1:
        faces[idx,0,0] = 32768.0
        return

    face_params[idx,0] = a
    face_params[idx,1] = b
    face_params[idx,2] = c
    face_params[idx,3] = -d

    cdef float x1,x2,x3,x4

    #detect intersection on the 12 edges of a box [-1000,1000]^3
    if abs(a)>=abs(b) and abs(a)>=abs(c):
        #x-direction (y,z) = (--,-+,++,+-)
        x1=-(b*(-border_limit)+c*(-border_limit)+d)/a
        x2=-(b*(-border_limit)+c*(border_limit)+d)/a
        x3=-(b*(border_limit)+c*(border_limit)+d)/a
        x4=-(b*(border_limit)+c*(-border_limit)+d)/a
        if a>0:
            faces[idx,0,0] = x1
            faces[idx,0,1] = -border_limit
            faces[idx,0,2] = -border_limit
            faces[idx,1,0] = x2
            faces[idx,1,1] = -border_limit
            faces[idx,1,2] = border_limit
            faces[idx,2,0] = x3
            faces[idx,2,1] = border_limit
            faces[idx,2,2] = border_limit
            faces[idx,3,0] = x4
            faces[idx,3,1] = border_limit
            faces[idx,3,2] = -border_limit
            faces[idx,4,0] = 32768.0
        else:
            faces[idx,3,0] = x1
            faces[idx,3,1] = -border_limit
            faces[idx,3,2] = -border_limit
            faces[idx,2,0] = x2
            faces[idx,2,1] = -border_limit
            faces[idx,2,2] = border_limit
            faces[idx,1,0] = x3
            faces[idx,1,1] = border_limit
            faces[idx,1,2] = border_limit
            faces[idx,0,0] = x4
            faces[idx,0,1] = border_limit
            faces[idx,0,2] = -border_limit
            faces[idx,4,0] = 32768.0
    elif abs(b)>=abs(c):
        #y-direction (x,z) = (--,-+,++,+-)
        x1=-(a*(-border_limit)+c*(-border_limit)+d)/b
        x2=-(a*(-border_limit)+c*(border_limit)+d)/b
        x3=-(a*(border_limit)+c*(border_limit)+d)/b
        x4=-(a*(border_limit)+c*(-border_limit)+d)/b
        if b<0:
            faces[idx,0,0] = -border_limit
            faces[idx,0,1] = x1
            faces[idx,0,2] = -border_limit
            faces[idx,1,0] = -border_limit
            faces[idx,1,1] = x2
            faces[idx,1,2] = border_limit
            faces[idx,2,0] = border_limit
            faces[idx,2,1] = x3
            faces[idx,2,2] = border_limit
            faces[idx,3,0] = border_limit
            faces[idx,3,1] = x4
            faces[idx,3,2] = -border_limit
            faces[idx,4,0] = 32768.0
        else:
            faces[idx,3,0] = -border_limit
            faces[idx,3,1] = x1
            faces[idx,3,2] = -border_limit
            faces[idx,2,0] = -border_limit
            faces[idx,2,1] = x2
            faces[idx,2,2] = border_limit
            faces[idx,1,0] = border_limit
            faces[idx,1,1] = x3
            faces[idx,1,2] = border_limit
            faces[idx,0,0] = border_limit
            faces[idx,0,1] = x4
            faces[idx,0,2] = -border_limit
            faces[idx,4,0] = 32768.0
    else:
        #z-direction (x,y) = (--,-+,++,+-)
        x1=-(a*(-border_limit)+b*(-border_limit)+d)/c
        x2=-(a*(-border_limit)+b*(border_limit)+d)/c
        x3=-(a*(border_limit)+b*(border_limit)+d)/c
        x4=-(a*(border_limit)+b*(-border_limit)+d)/c
        if c>0:
            faces[idx,0,0] = -border_limit
            faces[idx,0,1] = -border_limit
            faces[idx,0,2] = x1
            faces[idx,1,0] = -border_limit
            faces[idx,1,1] = border_limit
            faces[idx,1,2] = x2
            faces[idx,2,0] = border_limit
            faces[idx,2,1] = border_limit
            faces[idx,2,2] = x3
            faces[idx,3,0] = border_limit
            faces[idx,3,1] = -border_limit
            faces[idx,3,2] = x4
            faces[idx,4,0] = 32768.0
        else:
            faces[idx,3,0] = -border_limit
            faces[idx,3,1] = -border_limit
            faces[idx,3,2] = x1
            faces[idx,2,0] = -border_limit
            faces[idx,2,1] = border_limit
            faces[idx,2,2] = x2
            faces[idx,1,0] = border_limit
            faces[idx,1,1] = border_limit
            faces[idx,1,2] = x3
            faces[idx,0,0] = border_limit
            faces[idx,0,1] = -border_limit
            faces[idx,0,2] = x4
            faces[idx,4,0] = 32768.0

#put a plane into the mesh
#split faces if necessary
@cython.boundscheck(False)
@cython.wraparound(False)
def join_polygons(float[:, :, ::1] faces, float[:, ::1] face_params, char[::1] vtypes, Py_ssize_t idx):
    if faces[idx,0,0]>16384: return

    cdef float a,b,c,w

    a = face_params[idx,0]
    b = face_params[idx,1]
    c = face_params[idx,2]
    w = face_params[idx,3]

    cdef Py_ssize_t i,j,jj,j0,j1,pointer
    cdef char front_flag,back_flag
    cdef float dist,dist1,dist2,px,py,pz

    for i in range(idx):
        if faces[i,0,0]>16384: continue

        #split each face in face_group, if necessary
        #first detect whether split is needed
        front_flag = 0
        back_flag = 0
        j = 0
        while True:
            if faces[i,j,0]>16384: break
            dist = faces[i,j,0]*a + faces[i,j,1]*b + faces[i,j,2]*c - w
            if dist<-epsilon2: #back--2
                back_flag = 1
                vtypes[j] = 2
            elif dist>epsilon2: #front--1
                front_flag = 1
                vtypes[j] = 1
            else: #coplanar--0
                vtypes[j] = 0
            j += 1

        if front_flag and back_flag:
            #split
            #only save front part
            pointer = 0
            jj = 0
            j0 = 0
            j1 = 0
            faces[i,j,0] = faces[i,0,0]
            faces[i,j,1] = faces[i,0,1]
            faces[i,j,2] = faces[i,0,2]
            vtypes[j] = vtypes[0]
            faces[i,j+1,0] = faces[i,1,0]
            faces[i,j+1,1] = faces[i,1,1]
            faces[i,j+1,2] = faces[i,1,2]
            vtypes[j+1] = vtypes[1]
            faces[i,j+2,0] = faces[i,2,0]
            faces[i,j+2,1] = faces[i,2,1]
            faces[i,j+2,2] = faces[i,2,2]
            vtypes[j+2] = vtypes[2]
            while True:
                j0 = jj+2
                j1 = jj+3
                if vtypes[j0]!=2:
                    faces[i,pointer,0] = faces[i,j0,0]
                    faces[i,pointer,1] = faces[i,j0,1]
                    faces[i,pointer,2] = faces[i,j0,2]
                    pointer += 1
                if vtypes[j0]==1 and vtypes[j1]==2 or vtypes[j0]==2 and vtypes[j1]==1:
                    dist1 = faces[i,j0,0]*a + faces[i,j0,1]*b + faces[i,j0,2]*c
                    dist2 = faces[i,j1,0]*a + faces[i,j1,1]*b + faces[i,j1,2]*c
                    px = (w-dist1)*(faces[i,j1,0]-faces[i,j0,0])/(dist2-dist1)+faces[i,j0,0]
                    py = (w-dist1)*(faces[i,j1,1]-faces[i,j0,1])/(dist2-dist1)+faces[i,j0,1]
                    pz = (w-dist1)*(faces[i,j1,2]-faces[i,j0,2])/(dist2-dist1)+faces[i,j0,2]
                    faces[i,pointer,0] = px
                    faces[i,pointer,1] = py
                    faces[i,pointer,2] = pz
                    pointer += 1
                jj += 1
                if jj==j: break
            #add ending mark
            faces[i,pointer,0] = 32768.0
        elif back_flag:
            faces[i,0,0] = 32768.0
    
    
    #also split target_face
    for i in range(idx):
        if faces[i,0,0]>16384: continue

        #first detect whether split is needed
        a = face_params[i,0]
        b = face_params[i,1]
        c = face_params[i,2]
        w = face_params[i,3]
        front_flag = 0
        back_flag = 0
        j = 0
        while True:
            if faces[idx,j,0]>16384: break
            dist = faces[idx,j,0]*a + faces[idx,j,1]*b + faces[idx,j,2]*c - w
            if dist<-epsilon2: #back--2
                back_flag = 1
                vtypes[j] = 2
            elif dist>epsilon2: #front--1
                front_flag = 1
                vtypes[j] = 1
            else: #coplanar--0
                vtypes[j] = 0
            j += 1
        
        if front_flag and back_flag:
            #split
            #only save front part
            pointer = 0
            jj = 0
            j0 = 0
            j1 = 0
            faces[idx,j,0] = faces[idx,0,0]
            faces[idx,j,1] = faces[idx,0,1]
            faces[idx,j,2] = faces[idx,0,2]
            vtypes[j] = vtypes[0]
            faces[idx,j+1,0] = faces[idx,1,0]
            faces[idx,j+1,1] = faces[idx,1,1]
            faces[idx,j+1,2] = faces[idx,1,2]
            vtypes[j+1] = vtypes[1]
            faces[idx,j+2,0] = faces[idx,2,0]
            faces[idx,j+2,1] = faces[idx,2,1]
            faces[idx,j+2,2] = faces[idx,2,2]
            vtypes[j+2] = vtypes[2]
            while True:
                j0 = jj+2
                j1 = jj+3
                if vtypes[j0]!=2:
                    faces[idx,pointer,0] = faces[idx,j0,0]
                    faces[idx,pointer,1] = faces[idx,j0,1]
                    faces[idx,pointer,2] = faces[idx,j0,2]
                    pointer += 1
                if vtypes[j0]==1 and vtypes[j1]==2 or vtypes[j0]==2 and vtypes[j1]==1:
                    dist1 = faces[idx,j0,0]*a + faces[idx,j0,1]*b + faces[idx,j0,2]*c
                    dist2 = faces[idx,j1,0]*a + faces[idx,j1,1]*b + faces[idx,j1,2]*c
                    px = (w-dist1)*(faces[idx,j1,0]-faces[idx,j0,0])/(dist2-dist1)+faces[idx,j0,0]
                    py = (w-dist1)*(faces[idx,j1,1]-faces[idx,j0,1])/(dist2-dist1)+faces[idx,j0,1]
                    pz = (w-dist1)*(faces[idx,j1,2]-faces[idx,j0,2])/(dist2-dist1)+faces[idx,j0,2]
                    faces[idx,pointer,0] = px
                    faces[idx,pointer,1] = py
                    faces[idx,pointer,2] = pz
                    pointer += 1
                jj += 1
                if jj==j: break
            #add ending mark
            faces[idx,pointer,0] = 32768.0
        elif back_flag:
            faces[idx,0,0] = 32768.0
            break

#Union parametric faces to form a mesh, output vertices and polygons
#assume each face has <256-3 points
def digest_bsp(bsp_faces, float[:, :, ::1] faces, float[:, ::1] face_params, char[::1] vtypes, int bias):

    cdef Py_ssize_t i,j
    cdef float a,b,c,d
    cdef int bsp_faces_len,v_count

    bsp_faces_len = len(bsp_faces)

    #carve out the mesh face by face
    for i in range(bsp_faces_len):
        a,b,c,d = bsp_faces[i]
        get_polygon_from_params(faces, face_params, i, a,b,c,d)
        join_polygons(faces, face_params, vtypes, i)

    vertices = []
    polygons = []

    v_count = bias
    for i in range(bsp_faces_len):
        if faces[i,0,0]<16384:
            temp_face_idx = []
            for j in range(256):
                if faces[i,j,0]<16384:
                    vertices.append([faces[i,j,0],faces[i,j,1],faces[i,j,2]])
                    temp_face_idx.append(v_count)
                    v_count += 1
                else:
                    break
            polygons.append(temp_face_idx)
    
    return vertices, polygons


def get_mesh_watertight(bsp_convex_list):
    faces = np.full( [4096, 256, 3], 32768.0, np.float32 )
    face_params = np.full( [4096, 4], 32768.0, np.float32 )
    vtypes = np.full( [256], 32768, np.uint8 )
    vertices = []
    polygons = []
    merge_threshold = 1e-4

    for k in range(len(bsp_convex_list)):
        vg, tg = digest_bsp(bsp_convex_list[k],faces,face_params,vtypes,bias=0)
        biass = len(vertices)

        #merge same vertex
        mapping = np.zeros( [len(vg)], np.int32 )
        use_flag = np.zeros( [len(vg)], np.int32 )
        counter=0
        for i in range(len(vg)):
            same_flag = -1
            for j in range(i):
                if abs(vg[i][0]-vg[j][0])+abs(vg[i][1]-vg[j][1])+abs(vg[i][2]-vg[j][2])<merge_threshold:
                    same_flag = j
                    break
            if same_flag>0:
                mapping[i] = mapping[same_flag]
            else:
                mapping[i] = counter
                counter += 1
                use_flag[i] = True
        for i in range(len(vg)):
            if use_flag[i]:
                vertices.append(vg[i])
        for i in range(len(tg)):
            prev = mapping[tg[i][0]]
            tmpf = [prev+biass]
            for j in range(1,len(tg[i])):
                nowv = mapping[tg[i][j]]
                if nowv!=prev:
                    tmpf.append(nowv+biass)
                    prev = nowv
            if len(tmpf)>=3:
                polygons.append(tmpf)
    
    return vertices, polygons


def get_mesh(bsp_convex_list):
    faces = np.full( [4096, 256, 3], 32768.0, np.float32 )
    face_params = np.full( [4096, 4], 32768.0, np.float32 )
    vtypes = np.full( [256], 32768, np.uint8 )
    vertices = []
    polygons = []

    for k in range(len(bsp_convex_list)):
        vg, tg = digest_bsp(bsp_convex_list[k],faces,face_params,vtypes,bias=len(vertices))
        vertices = vertices+vg
        polygons = polygons+tg
    
    return vertices, polygons
