import numpy as np 
import math

def load_obj(dire):
    fin = open(dire,'r')
    lines = fin.readlines()
    fin.close()
    
    vertices = []
    triangles = []
    
    for i in range(len(lines)):
        line = lines[i].split()
        if len(line)==0:
            continue
        if line[0] == 'v':
            x = float(line[3])
            y = float(line[2])
            z = -float(line[1])
            vertices.append([x,y,z])
        if line[0] == 'f':
            x = int(line[1].split("/")[0])
            y = int(line[2].split("/")[0])
            z = int(line[3].split("/")[0])
            triangles.append([x-1,y-1,z-1])
    
    vertices = np.array(vertices, np.float32)
    
    
    #remove isolated points
    triangles_ = np.array(triangles, np.int32).reshape([-1])
    vertices_ = vertices[triangles_]
    
    
    #normalize diagonal=1
    x_max = np.max(vertices_[:,0])
    y_max = np.max(vertices_[:,1])
    z_max = np.max(vertices_[:,2])
    x_min = np.min(vertices_[:,0])
    y_min = np.min(vertices_[:,1])
    z_min = np.min(vertices_[:,2])
    
    x_mid = (x_max+x_min)/2
    y_mid = (y_max+y_min)/2
    z_mid = (z_max+z_min)/2
    
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    z_scale = z_max - z_min
    
    scale = math.sqrt(x_scale*x_scale + y_scale*y_scale + z_scale*z_scale)
    
    '''
    #normalize max=1
    x_max = np.max(vertices_[:,0])
    y_max = np.max(vertices_[:,1])
    z_max = np.max(vertices_[:,2])
    x_min = np.min(vertices_[:,0])
    y_min = np.min(vertices_[:,1])
    z_min = np.min(vertices_[:,2])
    
    x_mid = (x_max+x_min)/2
    y_mid = (y_max+y_min)/2
    z_mid = (z_max+z_min)/2
    
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    z_scale = z_max - z_min
    
    scale = max( max(x_scale, y_scale), z_scale)
    '''
    
    #print(len(vertices), len(triangles))
    vertices = np.array(vertices, np.float32)
    triangles = np.array(triangles, np.int32)
    
    
    vertices[:,0] = (vertices[:,0]-x_mid)/scale
    vertices[:,1] = (vertices[:,1]-y_mid)/scale
    vertices[:,2] = (vertices[:,2]-z_mid)/scale
    
    return vertices, triangles


def sample_points(vertices, triangles, num_of_points):
    epsilon = 1e-6
    triangle_area_list = np.zeros([len(triangles)],np.float32)
    triangle_normal_list = np.zeros([len(triangles),3],np.float32)
    for i in range(len(triangles)):
        #area = |u x v|/2 = |u||v|sin(uv)/2
        a,b,c = vertices[triangles[i,1]]-vertices[triangles[i,0]]
        x,y,z = vertices[triangles[i,2]]-vertices[triangles[i,0]]
        ti = b*z-c*y
        tj = c*x-a*z
        tk = a*y-b*x
        area2 = math.sqrt(ti*ti+tj*tj+tk*tk)
        if area2<epsilon:
            triangle_area_list[i] = 0
            triangle_normal_list[i,0] = 0
            triangle_normal_list[i,1] = 0
            triangle_normal_list[i,2] = 0
        else:
            triangle_area_list[i] = area2
            triangle_normal_list[i,0] = ti/area2
            triangle_normal_list[i,1] = tj/area2
            triangle_normal_list[i,2] = tk/area2
    
    triangle_area_sum = np.sum(triangle_area_list)
    sample_prob_list = (num_of_points/triangle_area_sum)*triangle_area_list

    triangle_index_list = np.arange(len(triangles))

    point_list = np.zeros([num_of_points,3],np.float32)
    normal_list = np.zeros([num_of_points,3],np.float32)
    count = 0
    watchdog = 0

    while(count<num_of_points):
        np.random.shuffle(triangle_index_list)
        watchdog += 1
        if watchdog>100:
            print("infinite loop here!")
            exit(0)
        for i in range(len(triangle_index_list)):
            if count>=num_of_points: break
            dxb = triangle_index_list[i]
            prob = sample_prob_list[dxb]
            prob_i = int(prob)
            prob_f = prob-prob_i
            if np.random.random()<prob_f:
                prob_i += 1
            normal_direction = triangle_normal_list[dxb]
            u = vertices[triangles[dxb,1]]-vertices[triangles[dxb,0]]
            v = vertices[triangles[dxb,2]]-vertices[triangles[dxb,0]]
            base = vertices[triangles[dxb,0]]
            for j in range(prob_i):
                #sample a point here:
                u_x = np.random.random()
                v_y = np.random.random()
                if u_x+v_y>=1:
                    u_x = 1-u_x
                    v_y = 1-v_y
                point_list[count] = u*u_x+v*v_y+base
                normal_list[count] = normal_direction
                count += 1
                if count>=num_of_points: break

    return point_list,normal_list

def read_off_triangle(shape_name):
    file = open(shape_name,'r')
    lines = file.readlines()

    line = lines[1].split()
    vertex_num = int(line[0])
    face_num = int(line[1])

    vertices = np.zeros([vertex_num,3], np.float32)
    triangles = np.zeros([face_num,3], np.int32)

    start = 2
    for i in range(vertex_num):
        line = lines[start].split()
        vertices[i,0] = float(line[2])
        vertices[i,1] = float(line[1])
        vertices[i,2] = -float(line[0])
        start += 1

    for i in range(face_num):
        line = lines[start].split()
        triangles[i,0] = int(line[1])
        triangles[i,1] = int(line[2])
        triangles[i,2] = int(line[3])
        start += 1

    return vertices, triangles

def read_ply_point(shape_name):
    file = open(shape_name,'r')
    lines = file.readlines()

    start = 0
    while True:
        line = lines[start].strip()
        if line == "end_header":
            start += 1
            break
        line = line.split()
        if line[0] == "element":
            if line[1] == "vertex":
                vertex_num = int(line[2])
        start += 1

    vertices = np.zeros([vertex_num,3], np.float32)
    for i in range(vertex_num):
        line = lines[i+start].split()
        vertices[i,0] = float(line[0]) #X
        vertices[i,1] = float(line[1]) #Y
        vertices[i,2] = float(line[2]) #Z
    return vertices

def read_ply_point_normal(shape_name):
    file = open(shape_name,'r')
    lines = file.readlines()

    start = 0
    while True:
        line = lines[start].strip()
        if line == "end_header":
            start += 1
            break
        line = line.split()
        if line[0] == "element":
            if line[1] == "vertex":
                vertex_num = int(line[2])
        start += 1

    vertices = np.zeros([vertex_num,3], np.float32)
    normals = np.zeros([vertex_num,3], np.float32)
    for i in range(vertex_num):
        line = lines[i+start].split()
        vertices[i,0] = float(line[0]) #X
        vertices[i,1] = float(line[1]) #Y
        vertices[i,2] = float(line[2]) #Z
        normals[i,0] = float(line[3]) #normalX
        normals[i,1] = float(line[4]) #normalY
        normals[i,2] = float(line[5]) #normalZ
    return vertices, normals

def read_ply_triangle(shape_name):
    file = open(shape_name,'r')
    lines = file.readlines()
    vertices = []
    triangles = []

    start = 0
    while True:
        line = lines[start].strip()
        if line == "end_header":
            start += 1
            break
        line = line.split()
        if line[0] == "element":
            if line[1] == "vertex":
                vertex_num = int(line[2])
            if line[1] == "face":
                face_num = int(line[2])
        start += 1

    vertices = np.zeros([vertex_num,3], np.float32)
    triangles = np.zeros([face_num,3], np.int32)

    for i in range(vertex_num):
        line = lines[start].split()
        vertices[i,0] = float(line[0])
        vertices[i,1] = float(line[1])
        vertices[i,2] = float(line[2])
        start += 1

    for i in range(face_num):
        line = lines[start].split()
        triangles[i,0] = int(line[1])
        triangles[i,1] = int(line[2])
        triangles[i,2] = int(line[3])
        start += 1

    return vertices, triangles

def write_ply_point(name, vertices):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    fout.close()


def write_ply_point_normal(name, vertices, normals=None):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property float nx\n")
    fout.write("property float ny\n")
    fout.write("property float nz\n")
    fout.write("end_header\n")
    if normals is None:
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(vertices[ii,3])+" "+str(vertices[ii,4])+" "+str(vertices[ii,5])+"\n")
    else:
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(normals[ii,0])+" "+str(normals[ii,1])+" "+str(normals[ii,2])+"\n")
    fout.close()


def write_ply_triangle(name, vertices, triangles):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face "+str(len(triangles))+"\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
    fout.close()


def write_ply_polygon(name, vertices, polygons):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face "+str(len(polygons))+"\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii][0])+" "+str(vertices[ii][1])+" "+str(vertices[ii][2])+"\n")
    for ii in range(len(polygons)):
        fout.write(str(len(polygons[ii])))
        for jj in range(len(polygons[ii])):
            fout.write(" "+str(polygons[ii][jj]))
        fout.write("\n")
    fout.close()


