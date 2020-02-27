import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import math
import random
import tensorflow as tf
import numpy as np
import cv2
import h5py

from read_write_etc import *


bspnet_point_dir = "/home/zhiqinc/zBSP/net/bsp_svr/samples/bsp_svr_out/"
edge_point_dir = "./test_edge/"


if not os.path.exists(edge_point_dir):
    os.makedirs(edge_point_dir)


eval_txt = open("all_vox256_img_test.txt","r")
eval_list = eval_txt.readlines()
eval_txt.close()
eval_list = [item.strip().split('/') for item in eval_list]
print(len(eval_list))

category_list = ['02691156_airplane','02828884_bench','02933112_cabinet','02958343_car','03001627_chair','03211117_display','03636649_lamp','03691459_speaker','04090263_rifle','04256520_couch','04379243_table','04401088_phone','04530566_vessel']
category_name = [name[:8] for name in category_list]
category_num = [809, 364, 315, 1500, 1356, 219, 464, 324, 475, 635, 1702, 211, 388]
category_num_sum = [809, 1173, 1488, 2988, 4344, 4563, 5027, 5351, 5826, 6461, 8163, 8374, 8762]
category_chamfer_distance_sum = [0.0]*13
category_normal_consistency_sum = [0.0]*13
category_count =[0]*13


run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True
with tf.Session(config=run_config) as sess:
    points_gt = tf.placeholder(shape=[None,3], dtype=tf.float32, name="points_gt")
    normals_gt = tf.placeholder(shape=[None,3], dtype=tf.float32, name="normals_gt")
    points_gtn = tf.stop_gradient(points_gt)
    normals_gtn = tf.stop_gradient(normals_gt)

    num_of_points = tf.shape(points_gt)[0]

    points_gt_mat1 = tf.tile(tf.reshape(points_gtn,[num_of_points,1,3]), [1,num_of_points,1])
    points_gt_mat2 = tf.tile(tf.reshape(points_gtn,[1,num_of_points,3]), [num_of_points,1,1])
    dist_gt = tf.reduce_sum(tf.square(points_gt_mat1-points_gt_mat2),axis=2)
    close_index_gt = tf.cast( dist_gt<0.0001 , tf.int8)

    normals_gt_mat1 = tf.tile(tf.reshape(normals_gtn,[num_of_points,1,3]), [1,num_of_points,1])
    normals_gt_mat2 = tf.tile(tf.reshape(normals_gtn,[1,num_of_points,3]), [num_of_points,1,1])
    prod_gt = tf.reduce_sum(normals_gt_mat1*normals_gt_mat2,axis=2)
    all_edge_index_gt = tf.cast( tf.abs(prod_gt)<0.1 , tf.int8)

    edge_index_gt = tf.reduce_max(close_index_gt*all_edge_index_gt, axis=1)

    for idx in range(len(eval_list)):
        class_name = eval_list[idx][0]
        object_name = eval_list[idx][1]
        print(class_name,object_name)

        #read bspnetnet 
        bspnet_ply_dir = bspnet_point_dir+str(idx)+'_pc.ply'
        bspnet_pc_vertices, bspnet_pc_normals = read_ply_point_normal(bspnet_ply_dir)

        #sanity check
        #write_ply_point_normal('gt.ply', gt_pc_vertices, gt_pc_normals)
        #write_ply_point_normal('pd.ply',  bspnet_pc_vertices, bspnet_pc_normals)
        #write_ply_triangle('pdm.ply', bspnet_ply_vertices, bspnet_ply_triangles)
        #exit(0)

        #compute Chamfer distance and Normal consistency
        edge_index = sess.run(edge_index_gt,
            feed_dict={
                points_gt: bspnet_pc_vertices,
                normals_gt: bspnet_pc_normals,
            })
        points = np.concatenate([bspnet_pc_vertices,bspnet_pc_normals],axis=1)
        points = points[edge_index>0.5]

        np.random.shuffle(points)

        write_ply_point_normal(edge_point_dir+str(idx)+'.ply',points[:4096])