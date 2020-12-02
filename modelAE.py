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

class BSP_AE(object):
	def __init__(self, sess, phase, sample_vox_size, is_training = False, ef_dim=32, c_dim=256, p_dim=4096, dataset_name='default', checkpoint_dir=None, sample_dir=None, data_dir='./data'):
		"""
		Args:
			too lazy to explain
		"""
		self.sess = sess
		self.phase = phase

		#progressive training
		#1-- (16, 16*16*16)
		#2-- (32, 16*16*16)
		#3-- (64, 16*16*16*4)
		self.sample_vox_size = sample_vox_size
		if self.sample_vox_size==16:
			self.point_batch_size = 16*16*16
			self.shape_batch_size = 36
		elif self.sample_vox_size==32:
			self.point_batch_size = 16*16*16
			self.shape_batch_size = 36
		elif self.sample_vox_size==64:
			self.point_batch_size = 16*16*16*4
			self.shape_batch_size = 12
		self.input_size = 64 #input voxel grid size

		self.p_dim = p_dim
		self.ef_dim = ef_dim
		self.c_dim = c_dim

		self.dataset_name = dataset_name
		self.dataset_load = dataset_name + '_train'
		if not is_training:
			self.dataset_load = dataset_name + '_test'
		self.checkpoint_dir = checkpoint_dir
		self.data_dir = data_dir
		
		data_hdf5_name = self.data_dir+'/'+self.dataset_load+'.hdf5'
		if os.path.exists(data_hdf5_name):
			data_dict = h5py.File(data_hdf5_name, 'r')
			self.data_points = data_dict['points_'+str(self.sample_vox_size)][:]
			self.data_values = data_dict['values_'+str(self.sample_vox_size)][:]
			self.data_voxels = data_dict['voxels'][:]
		else:
			print("error: cannot load "+data_hdf5_name)
			exit(0)
		
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
		self.vox3d = tf.placeholder(shape=[None,self.input_size,self.input_size,self.input_size,1], dtype=tf.float32, name="vox3d")
		self.point_coord = tf.placeholder(shape=[None,None,3], dtype=tf.float32, name="point_coord")
		self.point_value = tf.placeholder(shape=[None,None,1], dtype=tf.float32, name="point_value")
		self.plane_m = tf.placeholder(shape=[1,3,self.p_dim], dtype=tf.float32, name="plane_m")
		self.plane_b = tf.placeholder(shape=[1,1,self.p_dim], dtype=tf.float32, name="plane_b")
		self.convex_mask = tf.placeholder(shape=[1,1,self.c_dim], dtype=tf.float32, name="convex_mask")

		if phase==0:
			self.generator = self.generator0
		elif phase==1:
			self.generator = self.generator1_hard
		elif phase==2:
			self.generator = self.generator1_hard
		elif phase==3:
			self.generator = self.generator1_soft
		elif phase==4:
			self.generator = self.generator1_soft

		self.E_m, self.E_b, _ = self.encoder(self.vox3d, phase_train=True, reuse=False)
		self.G, _, self.G2, self.cw2, self.cw3 = self.generator(self.point_coord, self.E_m, self.E_b, phase_train=True, reuse=False)

		self.sE_m, self.sE_b, self.sE_z = self.encoder(self.vox3d, phase_train=False, reuse=True)
		self.zG, self.zG_max, self.zG2, _, _ = self.generator(self.point_coord, self.plane_m, self.plane_b, phase_train=False, reuse=True)
		self.zmG = tf.reduce_min(self.zG2+self.convex_mask, axis=2, keepdims=True)

		if phase==0:
			#phase 0 continuous for better convergence
			self.loss_sp = tf.reduce_mean(tf.square(self.point_value - self.G))
			self.loss = self.loss_sp + tf.reduce_sum(tf.abs(self.cw3-1)) + (tf.reduce_sum(tf.maximum(self.cw2-1,0)) - tf.reduce_sum(tf.minimum(self.cw2,0)))
			#L_recon + L_W + L_T
		elif phase==1:
			#phase 1 hard discrete for bsp
			self.loss_sp = tf.reduce_mean((1-self.point_value)*(1-tf.minimum(self.G,1)) + self.point_value*(tf.maximum(self.G,0)))
			self.loss = self.loss_sp
			#L_recon
		elif phase==2:
			#phase 2 hard discrete for bsp with L_overlap
			self.loss_sp = tf.reduce_mean((1-self.point_value)*(1-tf.minimum(self.G,1)) + self.point_value*(tf.maximum(self.G,0)))
			self.G2_inside = tf.cast(self.G2<0.01, tf.float32)
			self.bmask = self.G2_inside * tf.cast(tf.reduce_sum(self.G2_inside, axis=2, keepdims=True)>1, tf.float32)
			self.loss = self.loss_sp - tf.reduce_mean(self.G2*self.point_value*self.bmask)
			#L_recon + L_overlap
		elif phase==3:
			#phase 3 soft discrete for bsp
			self.loss_sp = tf.reduce_mean((1-self.point_value)*(1-tf.minimum(self.G,1)) + self.point_value*(tf.maximum(self.G,0)))
			self.loss = self.loss_sp + tf.reduce_sum(tf.abs(self.cw2)*tf.cast(self.cw2<0.01, tf.float32)) + tf.reduce_sum(tf.abs(self.cw2-1)*tf.cast(self.cw2>=0.01, tf.float32))
			#L_recon + L_T
			#soft cut with loss L_T: gradually move the values in T (self.cw2) to either 0 or 1
		elif phase==4:
			#phase 4 soft discrete for bsp with L_overlap
			self.loss_sp = tf.reduce_mean((1-self.point_value)*(1-tf.minimum(self.G,1)) + self.point_value*(tf.maximum(self.G,0)))
			self.G2_inside = tf.cast(self.G2<0.01, tf.float32)
			self.bmask = self.G2_inside * tf.cast(tf.reduce_sum(self.G2_inside, axis=2, keepdims=True)>1, tf.float32)
			self.loss = self.loss_sp + tf.reduce_sum(tf.abs(self.cw2)*tf.cast(self.cw2<0.01, tf.float32)) + tf.reduce_sum(tf.abs(self.cw2-1)*tf.cast(self.cw2>=0.01, tf.float32)) - tf.reduce_mean(self.G2*self.point_value*self.bmask)
			#L_recon + L_T + L_overlap
			#soft cut with loss L_T: gradually move the values in T (self.cw2) to either 0 or 1
			
		self.saver = tf.train.Saver(max_to_keep=2)
		
		
	def generator0(self, points, plane_m, plane_b, phase_train=True, reuse=False):
		with tf.variable_scope("simple_net") as scope:
			if reuse:
				scope.reuse_variables()
			
			#level 1
			h1 = tf.matmul(points, plane_m) + plane_b
			h1 = tf.maximum(h1, 0)
			
			#level 2
			convex_layer_weights = tf.get_variable("convex_layer_weights", [self.p_dim, self.c_dim], initializer=tf.random_normal_initializer(stddev=0.02))
			h2 = tf.matmul(h1, convex_layer_weights)
			h2 = tf.maximum(tf.minimum(1-h2, 1), 0)
			
			#level 3
			concave_layer_weights = tf.get_variable("concave_layer_weights", [self.c_dim, 1], initializer=tf.random_normal_initializer(stddev=0.02))
			h3 = tf.matmul(h2, concave_layer_weights)
			h3 = tf.maximum(tf.minimum(h3, 1), 0)
			h3_max = tf.reduce_max(h2, axis=2, keepdims=True)
			
		return h3, h3_max, h2, convex_layer_weights, concave_layer_weights

	def generator1_hard(self, points, plane_m, plane_b, phase_train=True, reuse=False):
		with tf.variable_scope("simple_net") as scope:
			if reuse:
				scope.reuse_variables()
			
			#level 1
			h1 = tf.matmul(points, plane_m) + plane_b
			h1 = tf.maximum(h1, 0)
			
			#level 2
			convex_layer_weights = tf.get_variable("convex_layer_weights", [self.p_dim, self.c_dim])
			convex_layer_weights = tf.cast(convex_layer_weights>0.01, convex_layer_weights.dtype)
			h2 = tf.matmul(h1, convex_layer_weights)
			
			#level 3
			h3 = tf.reduce_min(h2, axis=2, keepdims=True)
			h3_01 = tf.maximum(tf.minimum(1-tf.stop_gradient(h3), 1), 0)

		return h3, h3_01, h2, convex_layer_weights, None

	def generator1_soft(self, points, plane_m, plane_b, phase_train=True, reuse=False):
		with tf.variable_scope("simple_net") as scope:
			if reuse:
				scope.reuse_variables()
			
			#level 1
			h1 = tf.matmul(points, plane_m) + plane_b
			h1 = tf.maximum(h1, 0)
			
			#level 2
			convex_layer_weights = tf.get_variable("convex_layer_weights", [self.p_dim, self.c_dim])
			h2 = tf.matmul(h1, convex_layer_weights)
			
			#level 3
			h3 = tf.reduce_min(h2, axis=2, keepdims=True)
			h3_01 = tf.maximum(tf.minimum(1-tf.stop_gradient(h3), 1), 0)

		return h3, h3_01, h2, convex_layer_weights, None


	def encoder(self, inputs, phase_train=True, reuse=False):
		with tf.variable_scope("encoder") as scope:
			if reuse:
				scope.reuse_variables()
			
			d_1 = conv3d(inputs, shape=[4, 4, 4, 1, self.ef_dim], strides=[1,2,2,2,1], scope='conv_1')
			d_1 = lrelu(d_1)

			d_2 = conv3d(d_1, shape=[4, 4, 4, self.ef_dim, self.ef_dim*2], strides=[1,2,2,2,1], scope='conv_2')
			d_2 = lrelu(d_2)
			
			d_3 = conv3d(d_2, shape=[4, 4, 4, self.ef_dim*2, self.ef_dim*4], strides=[1,2,2,2,1], scope='conv_3')
			d_3 = lrelu(d_3)

			d_4 = conv3d(d_3, shape=[4, 4, 4, self.ef_dim*4, self.ef_dim*8], strides=[1,2,2,2,1], scope='conv_4')
			d_4 = lrelu(d_4)

			d_5 = conv3d(d_4, shape=[4, 4, 4, self.ef_dim*8, self.ef_dim*8], strides=[1,1,1,1,1], scope='conv_5', padding="VALID")
			d_5 = tf.reshape(d_5,[-1, self.ef_dim*8])
			d_5 = tf.nn.sigmoid(d_5)

			l1 = linear(d_5, self.ef_dim*16, scope='linear_1')
			l1 = lrelu(l1)

			l2 = linear(l1, self.ef_dim*32, scope='linear_2')
			l2 = lrelu(l2)

			l3 = linear(l2, self.ef_dim*64, scope='linear_3')
			l3 = lrelu(l3)

			l4_m = linear(l3, self.p_dim*3, scope='linear_4m')
			l4_b = linear(l3, self.p_dim, scope='linear_4b')

			l4_m = tf.reshape(l4_m,[-1, 3, self.p_dim])
			l4_b = tf.reshape(l4_b,[-1, 1, self.p_dim])

			return l4_m, l4_b, d_5
	
	def train(self, config):
		ae_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss)
		self.sess.run(tf.global_variables_initializer())
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			
		shape_num = len(self.data_voxels)
		batch_index_list = np.arange(shape_num)
		
		print("\n\n----------net summary----------")
		print("training samples   ", shape_num)
		print("-------------------------------\n\n")
		
		counter = 0
		start_time = time.time()
		assert config.epoch==0 or config.iteration==0
		training_epoch = config.epoch + int(config.iteration/shape_num)
		batch_num = int(shape_num/self.shape_batch_size)

		for epoch in range(0, training_epoch):
			np.random.shuffle(batch_index_list)
			avg_loss_sp = 0
			avg_loss_tt = 0
			avg_num = 0
			for idx in range(batch_num):
				dxb = batch_index_list[idx*self.shape_batch_size:(idx+1)*self.shape_batch_size]
				_, errSP, errTT = self.sess.run([ae_optim, self.loss_sp, self.loss],
					feed_dict={
						self.vox3d: self.data_voxels[dxb],
						self.point_coord: (self.data_points[dxb].astype(np.float32)+0.5)/256-0.5,
						self.point_value: self.data_values[dxb],
					})
				avg_loss_sp += errSP
				avg_loss_tt += errTT
				avg_num += 1
			print(str(self.sample_vox_size)+" Epoch: [%2d/%2d] time: %4.4f, loss_sp: %.6f, loss_total: %.6f" % (epoch, training_epoch, time.time() - start_time, avg_loss_sp/avg_num, avg_loss_tt/avg_num))
			if epoch%10==9:
				self.test_1(config,"train_"+str(self.sample_vox_size)+"_"+str(epoch))
			if epoch%20==19:
				self.save(config.checkpoint_dir, epoch)
				
		if config.phase==0:
			self.save(config.checkpoint_dir, self.sample_vox_size)
		else:
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
		
		t = np.random.randint(len(self.data_voxels))
		model_float = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2],np.float32)
		batch_voxels = self.data_voxels[t:t+1]
		out_m, out_b = self.sess.run([self.sE_m, self.sE_b],
			feed_dict={
				self.vox3d: batch_voxels,
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

		for t in range(config.start, min(len(self.data_voxels),config.end)):
			model_float = np.ones([self.real_size,self.real_size,self.real_size,self.c_dim],np.float32)
			batch_voxels = self.data_voxels[t:t+1]
			out_m, out_b = self.sess.run([self.sE_m, self.sE_b],
				feed_dict={
					self.vox3d: batch_voxels,
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
		for t in range(config.start, min(len(self.data_voxels),config.end)):
			print(t)
			model_float = np.ones([self.real_size,self.real_size,self.real_size,self.c_dim],np.float32)
			model_float_combined = np.ones([self.real_size,self.real_size,self.real_size],np.float32)
			batch_voxels = self.data_voxels[t:t+1]
			out_m, out_b = self.sess.run([self.sE_m, self.sE_b],
				feed_dict={
					self.vox3d: batch_voxels,
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


		for t in range(config.start, min(len(self.data_voxels),config.end)):
			model_float = np.ones([self.real_size,self.real_size,self.real_size,self.c_dim],np.float32)
			batch_voxels = self.data_voxels[t:t+1]
			out_m, out_b = self.sess.run([self.sE_m, self.sE_b],
				feed_dict={
					self.vox3d: batch_voxels,
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


	#output h3
	def test_dae3(self, config):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		dima = self.test_size
		dim = self.real_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier
		
		for t in range(config.start, min(len(self.data_voxels),config.end)):
			model_float = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2],np.float32)
			batch_voxels = self.data_voxels[t:t+1]
			out_m, out_b = self.sess.run([self.sE_m, self.sE_b],
				feed_dict={
					self.vox3d: batch_voxels,
				})
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						model_out = self.sess.run(self.zG,
							feed_dict={
								self.plane_m: out_m,
								self.plane_b: out_b,
								self.point_coord: self.coords[minib:minib+1],
							})
						model_float[self.aux_x+i+1,self.aux_y+j+1,self.aux_z+k+1] = np.reshape(model_out, [self.test_size,self.test_size,self.test_size])
			
			vertices, triangles = mcubes.marching_cubes(model_float, 0.5)
			vertices = (vertices-0.5)/self.real_size-0.5
			#output prediction
			write_ply_triangle(config.sample_dir+"/"+str(t)+"_vox.ply", vertices, triangles)

			vertices, triangles = mcubes.marching_cubes(batch_voxels[0,:,:,:,0], 0.5)
			vertices = (vertices-0.5)/self.real_size-0.5
			#output ground truth
			write_ply_triangle(config.sample_dir+"/"+str(t)+"_gt.ply", vertices, triangles)
			
			print("[sample]")
	
	def get_z(self, config):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return

		hdf5_path = self.checkpoint_dir+'/'+self.model_dir+'/'+self.dataset_name+'_train_z.hdf5'
		shape_num = len(self.data_voxels)
		hdf5_file = h5py.File(hdf5_path, mode='w')
		hdf5_file.create_dataset("zs", [shape_num,self.ef_dim*8], np.float32)

		print(shape_num)
		for idx in range(shape_num):
			batch_voxels = self.data_voxels[idx:idx+1]
			out_z = self.sess.run(self.sE_z,
				feed_dict={
					self.vox3d: batch_voxels,
				})
			hdf5_file["zs"][idx:idx+1,:] = out_z

		hdf5_file.close()
		print("[z]")
	
	@property
	def model_dir(self):
		return "{}_ae_{}".format(
				self.dataset_name, self.input_size)
			
	def save(self, checkpoint_dir, step):
		model_name = "BSP_AE.model"
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
