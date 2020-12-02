import os
import time
import math
import random
import tensorflow as tf
import numpy as np
import h5py
import mcubes
from bspt import digest_bsp, get_mesh, get_mesh_watertight
#from bspt_slow import digest_bsp, get_mesh, get_mesh_watertight

from ops import *
from utils import *

class BSP_SVR(object):
	def __init__(self, sess, phase, sample_vox_size, is_training = False, prev_ef_dim=32, ef_dim=64, c_dim=256, p_dim=4096, dataset_name='default', checkpoint_dir=None, sample_dir=None, data_dir='./data'):
		"""
		Args:
			too lazy to explain
		"""
		self.sess = sess

		#progressive training
		#1-- (16, 16*16*16)
		#2-- (32, 16*16*16)
		#3-- (64, 16*16*16*4)
		self.sample_vox_size = sample_vox_size
		if self.sample_vox_size==16:
			self.point_batch_size = 16*16*16
			self.shape_batch_size = 32
		elif self.sample_vox_size==32:
			self.point_batch_size = 16*16*16
			self.shape_batch_size = 32
		elif self.sample_vox_size==64:
			self.point_batch_size = 16*16*16*4
			self.shape_batch_size = 8
		self.input_size = 64 #input voxel grid size

		#actual batch size
		self.shape_batch_size = 64

		self.view_size = 137
		self.crop_size = 128
		self.view_num = 24
		self.crop_edge = self.view_size-self.crop_size
		self.test_idx = 23

		self.p_dim = p_dim
		self.ef_dim = ef_dim
		self.c_dim = c_dim
		self.prev_ef_dim = prev_ef_dim
		self.z_dim = prev_ef_dim*8

		self.dataset_name = dataset_name
		self.dataset_load = dataset_name + '_train'
		if not is_training:
			self.dataset_load = dataset_name + '_test'
		self.checkpoint_dir = checkpoint_dir
		self.data_dir = data_dir
		
		data_hdf5_name = self.data_dir+'/'+self.dataset_load+'.hdf5'
		if os.path.exists(data_hdf5_name):
			data_dict = h5py.File(data_hdf5_name, 'r')
			offset_x = int(self.crop_edge/2)
			offset_y = int(self.crop_edge/2)
			self.data_pixels = np.reshape(data_dict['pixels'][:,:,offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size], [-1,self.view_num,self.crop_size,self.crop_size,1])
		else:
			print("error: cannot load "+data_hdf5_name)
			exit(0)
		dataz_hdf5_name = self.checkpoint_dir+'/'+self.modelAE_dir+'/'+self.dataset_name+'_train_z.hdf5'
		if os.path.exists(dataz_hdf5_name):
			dataz_dict = h5py.File(dataz_hdf5_name, 'r')
			self.data_zs = dataz_dict['zs'][:]
		else:
			print("warning: cannot load "+dataz_hdf5_name)
		
		self.real_size = 64 #output point-value voxel grid size in testing
		self.test_size = 64 #related to testing batch_size, adjust according to gpu memory size
		test_point_batch_size = self.test_size*self.test_size*self.test_size #do not change

		#get coords
		dima = self.test_size
		dim = self.real_size
		self.aux_x = np.zeros([dima,dima,dima],np.uint8)
		self.aux_y = np.zeros([dima,dima,dima],np.uint8)
		self.aux_z = np.zeros([dima,dima,dima],np.uint8)
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier
		multiplier3 = multiplier*multiplier*multiplier
		for i in range(dima):
			for j in range(dima):
				for k in range(dima):
					self.aux_x[i,j,k] = i*multiplier
					self.aux_y[i,j,k] = j*multiplier
					self.aux_z[i,j,k] = k*multiplier
		self.coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
		for i in range(multiplier):
			for j in range(multiplier):
				for k in range(multiplier):
					self.coords[i*multiplier2+j*multiplier+k,:,:,:,0] = self.aux_x+i
					self.coords[i*multiplier2+j*multiplier+k,:,:,:,1] = self.aux_y+j
					self.coords[i*multiplier2+j*multiplier+k,:,:,:,2] = self.aux_z+k
		self.coords = (self.coords+0.5)/dim-0.5
		self.coords = np.reshape(self.coords,[multiplier3,test_point_batch_size,3])
	
		self.build_model(phase)

	def build_model(self, phase):
		#for train
		self.view = tf.placeholder(shape=[None,self.crop_size,self.crop_size,1], dtype=tf.float32, name="view")
		self.z_vector = tf.placeholder(shape=[None,self.z_dim], dtype=tf.float32, name="z_vector")
		
		#for test
		self.point_coord = tf.placeholder(shape=[1,None,3], dtype=tf.float32, name="point_coord")
		self.plane_m = tf.placeholder(shape=[1,3,self.p_dim], dtype=tf.float32, name="plane_m")
		self.plane_b = tf.placeholder(shape=[1,1,self.p_dim], dtype=tf.float32, name="plane_b")
		self.convex_mask = tf.placeholder(shape=[1,1,self.c_dim], dtype=tf.float32, name="convex_mask")
		
		self.E = self.img_encoder(self.view, phase_train=True, reuse=False)
		self.sE = self.img_encoder(self.view, phase_train=False, reuse=True)
		self.zE_m, self.zE_b = self.encoder(self.z_vector, phase_train=False, reuse=False)
		self.zG, self.zG_max, self.zG2, self.cw2, _ = self.generator(self.point_coord, self.plane_m, self.plane_b, phase_train=False, reuse=False)
		self.zmG = tf.reduce_min(self.zG2+self.convex_mask, axis=2, keepdims=True)

		self.vars = tf.trainable_variables()
		self.train_vars = [var for var in self.vars if 'img_encoder' in var.name]
		self.fixed_vars = [var for var in self.vars if 'img_encoder' not in var.name]

		self.loss = tf.reduce_mean(tf.square(self.z_vector - self.E))

		self.saver_ = tf.train.Saver(max_to_keep=10)
		self.saver = self.saver_

	def generator(self, points, plane_m, plane_b, phase_train=True, reuse=False):
		with tf.variable_scope("simple_net") as scope:
			if reuse:
				scope.reuse_variables()
			
			#level 1
			h1 = tf.matmul(points, plane_m) + plane_b
			h1 = tf.maximum(h1, 0)
			
			#level 2
			convex_layer_weights = tf.get_variable("convex_layer_weights", [self.p_dim, self.c_dim], initializer=tf.random_normal_initializer(stddev=0.02))
			convex_layer_weights = tf.cast(convex_layer_weights>0.01, convex_layer_weights.dtype)
			h2 = tf.matmul(h1, convex_layer_weights)
			
			#level 3
			h3 = tf.reduce_min(h2, axis=2, keepdims=True)
			h3_01 = tf.maximum(tf.minimum(1-tf.stop_gradient(h3), 1), 0)

		return h3, h3_01, h2, convex_layer_weights, None


	def img_encoder(self, view, phase_train=True, reuse=False):
		with tf.variable_scope("img_encoder") as scope:
			if reuse:
				scope.reuse_variables()
			
			#mimic resnet
			def resnet_block(input, dim_in, dim_out, scope):
				if dim_in == dim_out:
					output = conv2d_nobias(input, shape=[3, 3, dim_out, dim_out], strides=[1,1,1,1], scope=scope+'_1')
					output = lrelu(output)
					output = conv2d_nobias(output, shape=[3, 3, dim_out, dim_out], strides=[1,1,1,1], scope=scope+'_2')
					output = output + input
					output = lrelu(output)
				else:
					output = conv2d_nobias(input, shape=[3, 3, dim_in, dim_out], strides=[1,2,2,1], scope=scope+'_1')
					output = lrelu(output)
					output = conv2d_nobias(output, shape=[3, 3, dim_out, dim_out], strides=[1,1,1,1], scope=scope+'_2')
					input_ = conv2d_nobias(input, shape=[1, 1, dim_in, dim_out], strides=[1,2,2,1], scope=scope+'_3')
					output = output + input_
					output = lrelu(output)
				return output
			
			layer_0 = conv2d_nobias(1-view, shape=[7, 7, 1, self.ef_dim], strides=[1,2,2,1], scope='conv0')
			layer_0 = lrelu(layer_0)
			#no maxpool
			
			layer_1 = resnet_block(layer_0, self.ef_dim, self.ef_dim, 'conv1')
			layer_2 = resnet_block(layer_1, self.ef_dim, self.ef_dim, 'conv2')
			
			layer_3 = resnet_block(layer_2, self.ef_dim, self.ef_dim*2, 'conv3')
			layer_4 = resnet_block(layer_3, self.ef_dim*2, self.ef_dim*2, 'conv4')
			
			layer_5 = resnet_block(layer_4, self.ef_dim*2, self.ef_dim*4, 'conv5')
			layer_6 = resnet_block(layer_5, self.ef_dim*4, self.ef_dim*4, 'conv6')
			
			layer_7 = resnet_block(layer_6, self.ef_dim*4, self.ef_dim*8, 'conv7')
			layer_8 = resnet_block(layer_7, self.ef_dim*8, self.ef_dim*8, 'conv8')
			
			layer_9 = conv2d(layer_8, shape=[4, 4, self.ef_dim*8, self.ef_dim*16], strides=[1,2,2,1], scope='conv9')
			layer_9 = lrelu(layer_9)
			
			layer_10 = conv2d(layer_9, shape=[4, 4, self.ef_dim*16, self.ef_dim*16], strides=[1,1,1,1], scope='conv10', padding="VALID")
			layer_10 = tf.reshape(layer_10, [-1,self.ef_dim*16])
			layer_10 = lrelu(layer_10)

			l1 = linear(layer_10, self.ef_dim*16, scope='l1')
			l1 = lrelu(l1)

			l2 = linear(l1, self.ef_dim*16, scope='l2')
			l2 = lrelu(l2)

			l3 = linear(l2, self.ef_dim*16, scope='l3')
			l3 = lrelu(l3)

			l4 = linear(l3, self.z_dim, scope='l4')
			l4 = tf.nn.sigmoid(l4)

			return l4

	def encoder(self, zs, phase_train=True, reuse=False):
		with tf.variable_scope("encoder") as scope:
			if reuse:
				scope.reuse_variables()

			#d_5 = zs*self.z_normalizer
			l1 = linear(zs, self.prev_ef_dim*16, scope='linear_1')
			l1 = lrelu(l1)

			l2 = linear(l1, self.prev_ef_dim*32, scope='linear_2')
			l2 = lrelu(l2)

			l3 = linear(l2, self.prev_ef_dim*64, scope='linear_3')
			l3 = lrelu(l3)

			l4_m = linear(l3, self.p_dim*3, scope='linear_4m')
			l4_b = linear(l3, self.p_dim, scope='linear_4b')

			l4_m = tf.reshape(l4_m,[-1, 3, self.p_dim])
			l4_b = tf.reshape(l4_b,[-1, 1, self.p_dim])

			return l4_m, l4_b
	
	def train(self, config):
		#first time run
		svr_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss, var_list=self.train_vars)
		self.sess.run(tf.global_variables_initializer())
		
		self.saver = tf.train.Saver(self.fixed_vars)
		could_load, checkpoint_counter = self.loadAE(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			exit(-1)
		
		self.saver = self.saver_

		shape_num = len(self.data_pixels)
		batch_index_list = np.arange(shape_num)
		
		print("\n\n----------net summary----------")
		print("training samples   ", shape_num)
		print("-------------------------------\n\n")
		
		counter = 0
		start_time = time.time()
		assert config.epoch==0 or config.iteration==0
		training_epoch = config.epoch + int(config.iteration/shape_num)
		batch_num = int(shape_num/self.shape_batch_size)
		#batch_view = np.zeros([self.shape_batch_size,self.crop_size,self.crop_size,1], np.float32)

		for epoch in range(0, training_epoch):
			np.random.shuffle(batch_index_list)
			avg_loss = 0
			avg_num = 0
			for idx in range(batch_num):
				dxb = batch_index_list[idx*self.shape_batch_size:(idx+1)*self.shape_batch_size]

				'''
				#random flip - not used
				for t in range(self.shape_batch_size):
					which_view = np.random.randint(self.view_num)
					batch_view_ = self.data_pixels[dxb[t],which_view].astype(np.float32)
					if np.random.randint(2)==0:
						batch_view_ = np.flip(batch_view_, 1)
					batch_view[t] = batch_view_/255.0
				'''
				
				which_view = np.random.randint(self.view_num)
				batch_view = self.data_pixels[dxb,which_view].astype(np.float32)/255.0
				_, err = self.sess.run([svr_optim, self.loss],
					feed_dict={
						self.view: batch_view,
						self.z_vector: self.data_zs[dxb],
					})
				avg_loss += err
				avg_num += 1
			print(str(self.sample_vox_size)+" Epoch: [%2d/%2d] time: %4.4f, loss: %.8f" % (epoch, training_epoch, time.time() - start_time, avg_loss/avg_num))
			if epoch%10==9:
				self.test_1(config,"train_"+str(self.sample_vox_size)+"_"+str(epoch))
			if epoch%100==99:
				self.save(config.checkpoint_dir, epoch)

		self.save(config.checkpoint_dir, training_epoch)

	def test_1(self, config, name):
		multiplier = int(self.real_size/self.test_size)
		multiplier2 = multiplier*multiplier

		if config.phase==0:
			outG = self.zG
			thres = 0.5
		else:
			outG = self.zG_max
			thres = 0.99
		
		t = np.random.randint(len(self.data_pixels))
		model_float = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2],np.float32)
		batch_view = self.data_pixels[t:t+1,self.test_idx].astype(np.float32)/255.0
		out_z = self.sess.run(self.sE,
			feed_dict={
				self.view: batch_view,
			})
		out_m, out_b = self.sess.run([self.zE_m, self.zE_b],
			feed_dict={
				self.z_vector: out_z,
			})
		for i in range(multiplier):
			for j in range(multiplier):
				for k in range(multiplier):
					minib = i*multiplier2+j*multiplier+k
					model_out = self.sess.run(outG,
						feed_dict={
							self.plane_m: out_m,
							self.plane_b: out_b,
							self.point_coord: self.coords[minib:minib+1],
						})
					model_float[self.aux_x+i+1,self.aux_y+j+1,self.aux_z+k+1] = np.reshape(model_out, [self.test_size,self.test_size,self.test_size])
		
		vertices, triangles = mcubes.marching_cubes(model_float, thres)
		vertices = (vertices-0.5)/self.real_size-0.5
		write_ply_triangle(config.sample_dir+"/"+name+".ply", vertices, triangles)
		print("[sample]")


	#output bsp shape as ply
	def test_bsp(self, config):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		w2 = self.sess.run(self.cw2, feed_dict={})

		dima = self.test_size
		dim = self.real_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier

		for t in range(config.start, min(len(self.data_pixels),config.end)):
			model_float = np.ones([self.real_size,self.real_size,self.real_size,self.c_dim],np.float32)
			batch_view = self.data_pixels[t:t+1,self.test_idx].astype(np.float32)/255.0
			out_z = self.sess.run(self.sE,
				feed_dict={
					self.view: batch_view,
				})
			out_m, out_b = self.sess.run([self.zE_m, self.zE_b],
				feed_dict={
					self.z_vector: out_z,
				})
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						model_out = self.sess.run(self.zG2,
							feed_dict={
								self.plane_m: out_m,
								self.plane_b: out_b,
								self.point_coord: self.coords[minib:minib+1],
							})
						model_float[self.aux_x+i,self.aux_y+j,self.aux_z+k,:] = np.reshape(model_out, [self.test_size,self.test_size,self.test_size,self.c_dim])
			
			bsp_convex_list = []
			model_float = model_float<0.01
			model_float_sum = np.sum(model_float,axis=3)
			for i in range(self.c_dim):
				slice_i = model_float[:,:,:,i]
				if np.max(slice_i)>0: #if one voxel is inside a convex
					if np.min(model_float_sum-slice_i*2)>=0: #if this convex is redundant, i.e. the convex is inside the shape
						model_float_sum = model_float_sum-slice_i
					else:
						box = []
						for j in range(self.p_dim):
							if w2[j,i]>0.01:
								a = -out_m[0,0,j]
								b = -out_m[0,1,j]
								c = -out_m[0,2,j]
								d = -out_b[0,0,j]
								box.append([a,b,c,d])
						if len(box)>0:
							bsp_convex_list.append(np.array(box,np.float32))

			#print(bsp_convex_list)
			print(len(bsp_convex_list))
			
			#convert bspt to mesh
			#vertices, polygons = get_mesh(bsp_convex_list)
			#use the following alternative to merge nearby vertices to get watertight meshes
			vertices, polygons = get_mesh_watertight(bsp_convex_list)

			#output ply
			write_ply_polygon(config.sample_dir+"/"+str(t)+"_bsp.ply", vertices, polygons)
	
	#output bsp shape as ply and point cloud as ply
	def test_mesh_point(self, config):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
			
		w2 = self.sess.run(self.cw2, feed_dict={})
		dima = self.test_size
		dim = self.real_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier
		for t in range(config.start, min(len(self.data_pixels),config.end)):
			print(t)
			model_float = np.ones([self.real_size,self.real_size,self.real_size,self.c_dim],np.float32)
			model_float_combined = np.ones([self.real_size,self.real_size,self.real_size],np.float32)
			batch_view = self.data_pixels[t:t+1,self.test_idx].astype(np.float32)/255.0
			out_z = self.sess.run(self.sE,
				feed_dict={
					self.view: batch_view,
				})
			out_m, out_b = self.sess.run([self.zE_m, self.zE_b],
				feed_dict={
					self.z_vector: out_z,
				})
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						model_out, model_out_combined = self.sess.run([self.zG2, self.zG],
							feed_dict={
								self.plane_m: out_m,
								self.plane_b: out_b,
								self.point_coord: self.coords[minib:minib+1],
							})
						model_float[self.aux_x+i,self.aux_y+j,self.aux_z+k,:] = np.reshape(model_out, [self.test_size,self.test_size,self.test_size,self.c_dim])
						model_float_combined[self.aux_x+i,self.aux_y+j,self.aux_z+k] = np.reshape(model_out_combined, [self.test_size,self.test_size,self.test_size])

			# whether to use post processing to remove convexes that are inside the shape
			post_processing_flag = False
			
			if post_processing_flag:
				bsp_convex_list = []
				model_float = model_float<0.01
				model_float_sum = np.sum(model_float,axis=3)
				unused_convex = np.ones([self.c_dim], np.float32)
				for i in range(self.c_dim):
					slice_i = model_float[:,:,:,i]
					if np.max(slice_i)>0: #if one voxel is inside a convex
						if np.min(model_float_sum-slice_i*2)>=0: #if this convex is redundant, i.e. the convex is inside the shape
							model_float_sum = model_float_sum-slice_i
						else:
							box = []
							for j in range(self.p_dim):
								if w2[j,i]>0.01:
									a = -out_m[0,0,j]
									b = -out_m[0,1,j]
									c = -out_m[0,2,j]
									d = -out_b[0,0,j]
									box.append([a,b,c,d])
							if len(box)>0:
								bsp_convex_list.append(np.array(box,np.float32))
								unused_convex[i] = 0
								
				#convert bspt to mesh
				#vertices, polygons = get_mesh(bsp_convex_list)
				#use the following alternative to merge nearby vertices to get watertight meshes
				vertices, polygons = get_mesh_watertight(bsp_convex_list)

				#output ply
				write_ply_polygon(config.sample_dir+"/"+str(t)+"_bsp.ply", vertices, polygons)
				#output obj
				#write_obj_polygon(config.sample_dir+"/"+str(t)+"_bsp.obj", vertices, polygons)
				
				#sample surface points
				sampled_points_normals = sample_points_polygon(vertices, polygons, 16384)
				#check point inside shape or not
				sample_points_value = self.sess.run(self.zmG,
					feed_dict={
						self.plane_m: out_m,
						self.plane_b: out_b,
						self.convex_mask: np.reshape(unused_convex, [1,1,-1]),
						self.point_coord: np.reshape(sampled_points_normals[:,:3]+sampled_points_normals[:,3:]*1e-4, [1,-1,3]),
					})
				sampled_points_normals = sampled_points_normals[sample_points_value[0,:,0]>1e-4]
				print(len(bsp_convex_list), len(sampled_points_normals))
				np.random.shuffle(sampled_points_normals)
				write_ply_point_normal(config.sample_dir+"/"+str(t)+"_pc.ply", sampled_points_normals[:4096])
			else:
				bsp_convex_list = []
				model_float = model_float<0.01
				model_float_sum = np.sum(model_float,axis=3)
				for i in range(self.c_dim):
					slice_i = model_float[:,:,:,i]
					if np.max(slice_i)>0: #if one voxel is inside a convex
						#if np.min(model_float_sum-slice_i*2)>=0: #if this convex is redundant, i.e. the convex is inside the shape
						#	model_float_sum = model_float_sum-slice_i
						#else:
							box = []
							for j in range(self.p_dim):
								if w2[j,i]>0.01:
									a = -out_m[0,0,j]
									b = -out_m[0,1,j]
									c = -out_m[0,2,j]
									d = -out_b[0,0,j]
									box.append([a,b,c,d])
							if len(box)>0:
								bsp_convex_list.append(np.array(box,np.float32))
								
				#convert bspt to mesh
				#vertices, polygons = get_mesh(bsp_convex_list)
				#use the following alternative to merge nearby vertices to get watertight meshes
				vertices, polygons = get_mesh_watertight(bsp_convex_list)

				#output ply
				write_ply_polygon(config.sample_dir+"/"+str(t)+"_bsp.ply", vertices, polygons)
				#output obj
				#write_obj_polygon(config.sample_dir+"/"+str(t)+"_bsp.obj", vertices, polygons)
				
				#sample surface points
				sampled_points_normals = sample_points_polygon_vox64(vertices, polygons, model_float_combined, 16384)
				#check point inside shape or not
				sample_points_value = self.sess.run(self.zG,
					feed_dict={
						self.plane_m: out_m,
						self.plane_b: out_b,
						self.point_coord: np.reshape(sampled_points_normals[:,:3]+sampled_points_normals[:,3:]*1e-4, [1,-1,3]),
					})
				sampled_points_normals = sampled_points_normals[sample_points_value[0,:,0]>1e-4]
				print(len(bsp_convex_list), len(sampled_points_normals))
				np.random.shuffle(sampled_points_normals)
				write_ply_point_normal(config.sample_dir+"/"+str(t)+"_pc.ply", sampled_points_normals[:4096])

	#output bsp shape as obj with color
	def test_mesh_obj_material(self, config):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		w2 = self.sess.run(self.cw2, feed_dict={})

		dima = self.test_size
		dim = self.real_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier

		#write material
		#all output shapes share the same material
		#which means the same convex always has the same color for different shapes
		#change the colors in default.mtl to visualize correspondences between shapes
		fout2 = open(config.sample_dir+"/default.mtl", 'w')
		for i in range(self.c_dim):
			fout2.write("newmtl m"+str(i+1)+"\n") #material id
			fout2.write("Kd 0.80 0.80 0.80\n") #color (diffuse) RGB 0.00-1.00
			fout2.write("Ka 0 0 0\n") #color (ambient) leave 0s
		fout2.close()


		for t in range(config.start, min(len(self.data_pixels),config.end)):
			model_float = np.ones([self.real_size,self.real_size,self.real_size,self.c_dim],np.float32)
			batch_view = self.data_pixels[t:t+1,self.test_idx].astype(np.float32)/255.0
			out_z = self.sess.run(self.sE,
				feed_dict={
					self.view: batch_view,
				})
			out_m, out_b = self.sess.run([self.zE_m, self.zE_b],
				feed_dict={
					self.z_vector: out_z,
				})
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						model_out = self.sess.run(self.zG2,
							feed_dict={
								self.plane_m: out_m,
								self.plane_b: out_b,
								self.point_coord: self.coords[minib:minib+1],
							})
						model_float[self.aux_x+i,self.aux_y+j,self.aux_z+k,:] = np.reshape(model_out, [self.test_size,self.test_size,self.test_size,self.c_dim])
			
			bsp_convex_list = []
			color_idx_list = []
			model_float = model_float<0.01
			model_float_sum = np.sum(model_float,axis=3)
			for i in range(self.c_dim):
				slice_i = model_float[:,:,:,i]
				if np.max(slice_i)>0: #if one voxel is inside a convex
					if np.min(model_float_sum-slice_i*2)>=0: #if this convex is redundant, i.e. the convex is inside the shape
						model_float_sum = model_float_sum-slice_i
					else:
						box = []
						for j in range(self.p_dim):
							if w2[j,i]>0.01:
								a = -out_m[0,0,j]
								b = -out_m[0,1,j]
								c = -out_m[0,2,j]
								d = -out_b[0,0,j]
								box.append([a,b,c,d])
						if len(box)>0:
							bsp_convex_list.append(np.array(box,np.float32))
							color_idx_list.append(i)
			
			#print(bsp_convex_list)
			print(len(bsp_convex_list))
			
			#convert bspt to mesh
			vertices = []

			#write obj
			fout2 = open(config.sample_dir+"/"+str(t)+"_bsp.obj", 'w')
			fout2.write("mtllib default.mtl\n")

			for i in range(len(bsp_convex_list)):
				vg, tg = get_mesh([bsp_convex_list[i]])
				vbias=len(vertices)+1
				vertices = vertices+vg

				fout2.write("usemtl m"+str(color_idx_list[i]+1)+"\n")
				for ii in range(len(vg)):
					fout2.write("v "+str(vg[ii][0])+" "+str(vg[ii][1])+" "+str(vg[ii][2])+"\n")
				for ii in range(len(tg)):
					fout2.write("f")
					for jj in range(len(tg[ii])):
						fout2.write(" "+str(tg[ii][jj]+vbias))
					fout2.write("\n")

			fout2.close()

	@property
	def model_dir(self):
		return "{}_svr_{}".format(
				self.dataset_name, self.crop_size)
	@property
	def modelAE_dir(self):
		return "{}_ae_{}".format(
				self.dataset_name, self.input_size)

	def save(self, checkpoint_dir, step):
		model_name = "BSP_SVR.model"
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0

	def loadAE(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")
		checkpoint_dir = os.path.join(checkpoint_dir, self.modelAE_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0
