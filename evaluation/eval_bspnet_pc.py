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


gt_point_dir = "/local-scratch/zhiqinc/zBSP/sample_point_surface_4096/"
bspnet_point_dir = "/home/zhiqinc/zBSP/net/bsp_svr/samples/bsp_svr_out/"
num_of_points = 4096


eval_txt = open("all_vox256_img_test.txt","r")
eval_list = eval_txt.readlines()
eval_txt.close()
eval_list = [item.strip().split('/') for item in eval_list]
print(len(eval_list))

out_per_obj = open("result_per_obj.txt","w")
out_per_cat = open("result_per_category.txt","w")

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
    points_pd = tf.placeholder(shape=[None,3], dtype=tf.float32, name="points_pd")
    normals_pd = tf.placeholder(shape=[None,3], dtype=tf.float32, name="normals_pd")

    gt_pn = tf.shape(points_gt)[0]
    pd_pn = tf.shape(points_pd)[0]

    points_gt_mat = tf.tile(tf.reshape(points_gt,[gt_pn,1,3]), [1,pd_pn,1])
    points_pd_mat = tf.tile(tf.reshape(points_pd,[1,pd_pn,3]), [gt_pn,1,1])

    #distances
    dist = tf.reduce_sum(tf.square(points_gt_mat-points_pd_mat),axis=2)
    match_pd_gt = tf.argmin(dist, axis=0)
    match_gt_pd = tf.argmin(dist, axis=1)

    dist_pd_gt = tf.reduce_mean(tf.square(points_pd - tf.gather(points_gt,match_pd_gt)))*3
    dist_gt_pd = tf.reduce_mean(tf.square(points_gt - tf.gather(points_pd,match_gt_pd)))*3

    # Handle normals that point into wrong direction gracefully
    # (mostly due to mehtod not caring about this in generation)
    normals_dot_pd_gt = tf.reduce_mean(tf.abs(tf.reduce_sum(normals_pd*tf.gather(normals_gt,match_pd_gt), axis=1)))
    normals_dot_gt_pd = tf.reduce_mean(tf.abs(tf.reduce_sum(normals_gt*tf.gather(normals_pd,match_gt_pd), axis=1)))

    chamfer_distance = dist_pd_gt+dist_gt_pd
    normal_consistency = (normals_dot_pd_gt+normals_dot_gt_pd)/2

    for idx in range(len(eval_list)):
        class_name = eval_list[idx][0]
        object_name = eval_list[idx][1]
        print(class_name,object_name)

        #read ground truth point cloud
        gt_pc_dir = gt_point_dir+class_name+'/'+object_name+'.ply'
        gt_pc_vertices, gt_pc_normals = read_ply_point_normal(gt_pc_dir)

        #read bspnet
		bspnet_pc_dir = bspnet_point_dir+str(idx)+'_pc.ply'
        bspnet_pc_vertices, bspnet_pc_normals = read_ply_point_normal(bspnet_pc_dir)

        #sanity check
        #write_ply_point_normal('gt.ply', gt_pc_vertices, gt_pc_normals)
        #write_ply_point_normal('pd.ply', bspnet_pc_vertices, bspnet_pc_normals)
        #exit(0)

        #compute Chamfer distance and Normal consistency
        chamfer_distance_out, normal_consistency_out = sess.run([chamfer_distance, normal_consistency],
            feed_dict={
                points_gt: gt_pc_vertices,
                normals_gt: gt_pc_normals,
                points_pd: bspnet_pc_vertices,
                normals_pd: bspnet_pc_normals,
            })
        
        print(idx, chamfer_distance_out, normal_consistency_out)
        cat_id = category_name.index(class_name)
        category_count[cat_id] += 1
        category_chamfer_distance_sum[cat_id] += chamfer_distance_out
        category_normal_consistency_sum[cat_id] += normal_consistency_out
        out_per_obj.write( str(chamfer_distance_out)+' '+str(normal_consistency_out)+'\n' )

for i in range(13):
    out_per_cat.write(str(category_chamfer_distance_sum[i]/category_count[i]))
    out_per_cat.write('\t')
out_per_cat.write('\n')
for i in range(13):
    out_per_cat.write(str(category_normal_consistency_sum[i]/category_count[i]))
    out_per_cat.write('\t')
out_per_cat.write('\n')


out_per_obj.close()
out_per_cat.close()