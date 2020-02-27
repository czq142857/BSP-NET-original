import numpy as np
from math import sqrt
#2D version
#compatible version - slow, please use the Cython implementation if inference speed is of importance

border_limit = 1.0

#convert a parametric face to a polygon
def get_polygon_from_params(params):
    epsilon = 1e-5
    face = []
    a,b,d = params
    sum = a*a+b*b
    if sum<epsilon:
        return None

    #detect intersection on the 12 edges of a box [-1000,1000]^3
    if abs(a)>=abs(b):
        #x-direction (y) = (-,+)
        x1=-(b*(-border_limit)+d)/a
        x3=-(b*(border_limit)+d)/a
        face.append([a,b,-d])
        if a>0:
            face.append(np.array([x1,-border_limit]))
            face.append(np.array([x3,border_limit]))
        else:
            face.append(np.array([x3,border_limit]))
            face.append(np.array([x1,-border_limit]))
    else:
        #y-direction (x,z) = (-,+)
        y1=-(a*(-border_limit)+d)/b
        y3=-(a*(border_limit)+d)/b
        face.append([a,b,-d])
        if b<0:
            face.append(np.array([-border_limit,y1]))
            face.append(np.array([border_limit,y3]))
        else:
            face.append(np.array([border_limit,y3]))
            face.append(np.array([-border_limit,y1]))

    return face

#put a plane into the mesh
#split faces if necessary
def join_polygons(target_face,face_group):
    epsilon = 1e-5
    faces = []
    a,b,w = target_face[0]
    
    for i in range(len(face_group)):
        #split each face in face_group, if necessary
        #first detect whether split is needed
        face_i = face_group[i]
        front_flag = False
        back_flag = False
        vtypes = [-1] #first element is a dummy
        for j in range(1,len(face_i)):
            dist = face_i[j][0]*a + face_i[j][1]*b - w
            if dist<-epsilon: #back--2
                back_flag = True
                vtypes.append(2)
            elif dist>epsilon: #front--1
                front_flag = True
                vtypes.append(1)
            else: #coplanar--0
                vtypes.append(0)
        
        if front_flag and back_flag:
            #split
            #only save front part
            face_i_new = [face_i[0]]
            j = 1
            j1 = j+1
            if vtypes[j]!=2:
                face_i_new.append(face_i[j])
            if vtypes[j]==1 and vtypes[j1]==2 or vtypes[j]==2 and vtypes[j1]==1:
                dist1 = face_i[j][0]*a + face_i[j][1]*b
                dist2 = face_i[j1][0]*a + face_i[j1][1]*b
                p = (w-dist1)*(face_i[j1]-face_i[j])/(dist2-dist1)+face_i[j]
                face_i_new.append(p)
            if vtypes[j1]!=2:
                face_i_new.append(face_i[j1])
            faces.append(face_i_new)
        elif front_flag:
            faces.append(face_i)
    
    
    #also split target_face
    onsurface_flag = True
    result_face = []
    for k in range(len(target_face)):
        result_face.append(target_face[k])
    
    for i in range(len(face_group)):
        #first detect whether split is needed
        face_i = face_group[i]
        a,b,w = face_i[0]
        front_flag = False
        back_flag = False
        vtypes = [-1] #first element is a dummy
        for j in range(1,len(result_face)):
            dist = result_face[j][0]*a + result_face[j][1]*b - w
            if dist<-epsilon: #back--2
                back_flag = True
                vtypes.append(2)
            elif dist>epsilon: #front--1
                front_flag = True
                vtypes.append(1)
            else: #coplanar--0
                vtypes.append(0)
        
        if front_flag and back_flag:
            #split
            #only save front part
            result_face_new = [result_face[0]]
            j = 1
            j1 = j+1
            if vtypes[j]!=2:
                result_face_new.append(result_face[j])
            if vtypes[j]==1 and vtypes[j1]==2 or vtypes[j]==2 and vtypes[j1]==1:
                dist1 = result_face[j][0]*a + result_face[j][1]*b
                dist2 = result_face[j1][0]*a + result_face[j1][1]*b
                p = (w-dist1)*(result_face[j1]-result_face[j])/(dist2-dist1)+result_face[j]
                result_face_new.append(p)
            if vtypes[j1]!=2:
                result_face_new.append(result_face[j1])
            result_face = result_face_new
        elif back_flag:
            onsurface_flag = False
            break
        
    if onsurface_flag:
        faces.append(result_face)
    return faces

#Union parametric faces to form a mesh, output vertices and polygons
def digest_bsp(bsp_convex,bias):
    faces = []

    #carve out the mesh face by face
    for i in range(len(bsp_convex)):
        temp_face = get_polygon_from_params(bsp_convex[i])
        if temp_face is not None:
            faces = join_polygons(temp_face,faces)

    vertices = []
    polygons = []

    #add "merge same vertex" in the future?
    v_count = bias
    for i in range(len(faces)):
        temp_face_idx = []
        for j in range(1,len(faces[i])):
            vertices.append(faces[i][j])
            temp_face_idx.append(v_count)
            v_count += 1
        polygons.append(temp_face_idx)
    
    return vertices, polygons

def get_mesh(bsp_convex_list):
    vertices = []
    polygons = []

    for i in range(len(bsp_convex_list)):
        vg, tg = digest_bsp(bsp_convex_list[i],bias=len(vertices))
        vertices = vertices+vg
        polygons = polygons+tg
    
    return vertices, polygons
