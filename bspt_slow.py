import numpy as np
from math import sqrt

#input format
#convex_list -> [convex,convex,...]
#convex -> [face,face,...]
#face -> [a,b,c,d]
#ax+by+cz+d=0 is the plane
#ax+by+cz+d>0 is the inner part

#generate 18-face-shape
def ballike(x,y,z,a,b,c):
    v_center = np.array([-x,-y,-z,1],np.float32)
    v_span = np.array([1/a,1/b,1/c,1],np.float32)
    box = np.array([
        #face
        [-1,0,0,0.5],
        [1,0,0,0.5],
        [0,-1,0,0.5],
        [0,1,0,0.5],
        [0,0,-1,0.5],
        [0,0,1,0.5],
        #edge-x
        [0,-1,1,0.75],
        [0,1,1,0.75],
        [0,-1,-1,0.75],
        [0,1,-1,0.75],
        #edge-y
        [-1,0,1,0.75],
        [1,0,1,0.75],
        [-1,0,-1,0.75],
        [1,0,-1,0.75],
        #edge-z
        [-1,1,0,0.75],
        [1,1,0,0.75],
        [-1,-1,0,0.75],
        [1,-1,0,0.75],

    ],np.float32)

    box = box*v_span
    for i in range(len(box)):
        box[i,3] = np.sum(box[i]*v_center)

    return box

bsp_convex_list =[
    ballike(0,0,0,0.1,0.1,0.1),
    ballike(0.2,0,0,0.15,0.1,0.1),
    ballike(0,0.2,0,0.1,0.15,0.1),
    ballike(0,0,0.2,0.1,0.15,0.2),
]


#note: assume the mesh is convex and within [-1,1]^3
#face: [ [nx,ny,nz,w], v1,v2,v3,... ]
#nx,ny,nz is normal = a,b,c
#w = dot(normal,v1) = -d distance from origin to the plane
border_limit = 1.0

#convert a parametric face to a polygon
def get_polygon_from_params(params):
    epsilon = 1e-5
    face = []
    a,b,c,d = params
    sum = a*a+b*b+c*c
    if sum<epsilon:
        return None

    #detect intersection on the 12 edges of a box [-1000,1000]^3
    if abs(a)>=abs(b) and abs(a)>=abs(c):
        #x-direction (y,z) = (--,-+,++,+-)
        x1=-(b*(-border_limit)+c*(-border_limit)+d)/a
        x2=-(b*(-border_limit)+c*(border_limit)+d)/a
        x3=-(b*(border_limit)+c*(border_limit)+d)/a
        x4=-(b*(border_limit)+c*(-border_limit)+d)/a
        face.append([a,b,c,-d])
        if a>0:
            face.append(np.array([x1,-border_limit,-border_limit]))
            face.append(np.array([x2,-border_limit,border_limit]))
            face.append(np.array([x3,border_limit,border_limit]))
            face.append(np.array([x4,border_limit,-border_limit]))
        else:
            face.append(np.array([x4,border_limit,-border_limit]))
            face.append(np.array([x3,border_limit,border_limit]))
            face.append(np.array([x2,-border_limit,border_limit]))
            face.append(np.array([x1,-border_limit,-border_limit]))
    elif abs(b)>=abs(c):
        #y-direction (x,z) = (--,-+,++,+-)
        y1=-(a*(-border_limit)+c*(-border_limit)+d)/b
        y2=-(a*(-border_limit)+c*(border_limit)+d)/b
        y3=-(a*(border_limit)+c*(border_limit)+d)/b
        y4=-(a*(border_limit)+c*(-border_limit)+d)/b
        face.append([a,b,c,-d])
        if b<0:
            face.append(np.array([-border_limit,y1,-border_limit]))
            face.append(np.array([-border_limit,y2,border_limit]))
            face.append(np.array([border_limit,y3,border_limit]))
            face.append(np.array([border_limit,y4,-border_limit]))
        else:
            face.append(np.array([border_limit,y4,-border_limit]))
            face.append(np.array([border_limit,y3,border_limit]))
            face.append(np.array([-border_limit,y2,border_limit]))
            face.append(np.array([-border_limit,y1,-border_limit]))
    else:
        #z-direction (x,y) = (--,-+,++,+-)
        z1=-(a*(-border_limit)+b*(-border_limit)+d)/c
        z2=-(a*(-border_limit)+b*(border_limit)+d)/c
        z3=-(a*(border_limit)+b*(border_limit)+d)/c
        z4=-(a*(border_limit)+b*(-border_limit)+d)/c
        face.append([a,b,c,-d])
        if c>0:
            face.append(np.array([-border_limit,-border_limit,z1]))
            face.append(np.array([-border_limit,border_limit,z2]))
            face.append(np.array([border_limit,border_limit,z3]))
            face.append(np.array([border_limit,-border_limit,z4]))
        else:
            face.append(np.array([border_limit,-border_limit,z4]))
            face.append(np.array([border_limit,border_limit,z3]))
            face.append(np.array([-border_limit,border_limit,z2]))
            face.append(np.array([-border_limit,-border_limit,z1]))

    return face

#put a plane into the mesh
#split faces if necessary
def join_polygons(target_face,face_group):
    epsilon = 1e-5
    faces = []
    a,b,c,w = target_face[0]
    
    for i in range(len(face_group)):
        #split each face in face_group, if necessary
        #first detect whether split is needed
        face_i = face_group[i]
        front_flag = False
        back_flag = False
        vtypes = [-1] #first element is a dummy
        for j in range(1,len(face_i)):
            dist = face_i[j][0]*a + face_i[j][1]*b + face_i[j][2]*c - w
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
            for j in range(1,len(face_i)):
                j1 = j+1
                if j1==len(face_i):
                    j1=1
                if vtypes[j]!=2:
                    face_i_new.append(face_i[j])
                if vtypes[j]==1 and vtypes[j1]==2 or vtypes[j]==2 and vtypes[j1]==1:
                    dist1 = face_i[j][0]*a + face_i[j][1]*b + face_i[j][2]*c
                    dist2 = face_i[j1][0]*a + face_i[j1][1]*b + face_i[j1][2]*c
                    p = (w-dist1)*(face_i[j1]-face_i[j])/(dist2-dist1)+face_i[j]
                    
                    dist1 = target_face[1][0]*a + target_face[1][1]*b + target_face[1][2]*c
                    dist2 = target_face[2][0]*a + target_face[2][1]*b + target_face[2][2]*c
                    dist3 = target_face[3][0]*a + target_face[3][1]*b + target_face[3][2]*c
                    dist4 = p[0]*a + p[1]*b + p[2]*c
                    face_i_new.append(p)
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
        a,b,c,w = face_i[0]
        front_flag = False
        back_flag = False
        vtypes = [-1] #first element is a dummy
        for j in range(1,len(result_face)):
            dist = result_face[j][0]*a + result_face[j][1]*b + result_face[j][2]*c - w
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
            for j in range(1,len(result_face)):
                j1 = j+1
                if j1==len(result_face):
                    j1=1
                if vtypes[j]!=2:
                    result_face_new.append(result_face[j])
                if vtypes[j]==1 and vtypes[j1]==2 or vtypes[j]==2 and vtypes[j1]==1:
                    dist1 = result_face[j][0]*a + result_face[j][1]*b + result_face[j][2]*c
                    dist2 = result_face[j1][0]*a + result_face[j1][1]*b + result_face[j1][2]*c
                    p = (w-dist1)*(result_face[j1]-result_face[j])/(dist2-dist1)+result_face[j]
                    result_face_new.append(p)
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

#this function can handle empty outputs, 'digest_bsp' cannot.
def digest_bsp_robust(bsp_convex,bias):
    faces = [
        [[-1.0, 0.0, 0.0, border_limit], np.array([ border_limit, -border_limit,  border_limit]), np.array([ border_limit, -border_limit, -border_limit]), np.array([ border_limit,  border_limit, -border_limit]), np.array([border_limit, border_limit, border_limit])], 
        [[1.0, 0.0, 0.0, border_limit], np.array([-border_limit,  border_limit,  border_limit]), np.array([-border_limit,  border_limit, -border_limit]), np.array([-border_limit, -border_limit, -border_limit]), np.array([-border_limit, -border_limit,  border_limit])], 
        [[0.0, -1.0, 0.0, border_limit], np.array([border_limit, border_limit, border_limit]), np.array([ border_limit,  border_limit, -border_limit]), np.array([-border_limit,  border_limit, -border_limit]), np.array([-border_limit,  border_limit,  border_limit])], 
        [[0.0, 1.0, 0.0, border_limit], np.array([-border_limit, -border_limit,  border_limit]), np.array([-border_limit, -border_limit, -border_limit]), np.array([ border_limit, -border_limit, -border_limit]), np.array([ border_limit, -border_limit,  border_limit])], 
        [[0.0, 0.0, -1.0, border_limit], np.array([-border_limit,  border_limit,  border_limit]), np.array([-border_limit, -border_limit,  border_limit]), np.array([ border_limit, -border_limit,  border_limit]), np.array([border_limit, border_limit, border_limit])], 
        [[0.0, 0.0, 1.0, border_limit], np.array([ border_limit,  border_limit, -border_limit]), np.array([ border_limit, -border_limit, -border_limit]), np.array([-border_limit, -border_limit, -border_limit]), np.array([-border_limit,  border_limit, -border_limit])]
    ]

    #carve out the mesh face by face
    for i in range(len(bsp_convex)):
        temp_face = get_polygon_from_params(bsp_convex[i])
        if temp_face is not None:
            faces = join_polygons(temp_face,faces)
            if len(faces)<=1:
                return [],[]

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

def get_mesh_watertight(bsp_convex_list):
    vertices = []
    polygons = []
    merge_threshold = 1e-4

    for k in range(len(bsp_convex_list)):
        vg, tg = digest_bsp(bsp_convex_list[k],bias=0)
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
            if same_flag>=0:
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
    vertices = []
    polygons = []

    for i in range(len(bsp_convex_list)):
        vg, tg = digest_bsp(bsp_convex_list[i],bias=len(vertices))
        vertices = vertices+vg
        polygons = polygons+tg
    
    return vertices, polygons




if __name__=='__main__':
    vertices, polygons = get_mesh(bsp_convex_list)

    #output ply
    fout = open("mesh.ply", 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face "+str(len(polygons))+"\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")

    for i in range(len(vertices)):
        fout.write(str(vertices[i][0])+" "+str(vertices[i][1])+" "+str(vertices[i][2])+"\n")

    for i in range(len(polygons)):
        fout.write(str(len(polygons[i])))
        for j in range(len(polygons[i])):
            fout.write(" "+str(polygons[i][j]))
        fout.write("\n")

    fout.close()

