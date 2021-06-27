import os
import time
import math
import random
import tensorflow as tf
import numpy as np
import h5py
import mcubes
import cv2

from ops import *
from utils import *

class BSP_AE(object):
	def __init__(self, sess, sample_vox_size, is_training = False, ef_dim=32, c_dim=256, p_dim=2048, dataset_name='default', checkpoint_dir=None, sample_dir=None, data_dir='./data'):
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
		
		if is_training:
			data_hdf5_name = self.data_dir+'/'+self.dataset_load+'.hdf5'
			if os.path.exists(data_hdf5_name):
				data_dict = h5py.File(data_hdf5_name, 'r')
				self.data_points = data_dict['points_'+str(self.sample_vox_size)][:]
				self.data_values = data_dict['values_'+str(self.sample_vox_size)][:]
				self.data_voxels = data_dict['voxels'][:]
			else:
				print("error: cannot load "+data_hdf5_name)
				exit(0)
		
		#keep everything a power of 2
		self.cell_grid_size = 4
		self.frame_grid_size = 64
		self.real_size = self.cell_grid_size*self.frame_grid_size #=256, output point-value voxel grid size in testing
		self.test_point_num = 16*16*16*4*4 #adjust test_point_num according to gpu memory size in testing
		
		#get coords
		dimc = self.cell_grid_size
		dimf = self.frame_grid_size
		self.cell_x = np.zeros([dimc,dimc,dimc],np.int32)
		self.cell_y = np.zeros([dimc,dimc,dimc],np.int32)
		self.cell_z = np.zeros([dimc,dimc,dimc],np.int32)
		self.cell_coords = np.zeros([dimf,dimf,dimf,dimc,dimc,dimc,3],np.float32)
		self.frame_coords = np.zeros([dimf,dimf,dimf,3],np.float32)
		self.frame_x = np.zeros([dimf,dimf,dimf],np.int32)
		self.frame_y = np.zeros([dimf,dimf,dimf],np.int32)
		self.frame_z = np.zeros([dimf,dimf,dimf],np.int32)
		for i in range(dimc):
			for j in range(dimc):
				for k in range(dimc):
					self.cell_x[i,j,k] = i
					self.cell_y[i,j,k] = j
					self.cell_z[i,j,k] = k
		for i in range(dimf):
			for j in range(dimf):
				for k in range(dimf):
					self.cell_coords[i,j,k,:,:,:,0] = self.cell_x+i*dimc
					self.cell_coords[i,j,k,:,:,:,1] = self.cell_y+j*dimc
					self.cell_coords[i,j,k,:,:,:,2] = self.cell_z+k*dimc
					self.frame_coords[i,j,k,0] = i
					self.frame_coords[i,j,k,1] = j
					self.frame_coords[i,j,k,2] = k
					self.frame_x[i,j,k] = i
					self.frame_y[i,j,k] = j
					self.frame_z[i,j,k] = k
		self.cell_coords = (self.cell_coords+0.5)/self.real_size-0.5
		self.cell_coords = np.reshape(self.cell_coords,[dimf,dimf,dimf,dimc*dimc*dimc,3])
		self.cell_x = np.reshape(self.cell_x,[dimc*dimc*dimc])
		self.cell_y = np.reshape(self.cell_y,[dimc*dimc*dimc])
		self.cell_z = np.reshape(self.cell_z,[dimc*dimc*dimc])
		self.frame_x = np.reshape(self.frame_x,[dimf*dimf*dimf])
		self.frame_y = np.reshape(self.frame_y,[dimf*dimf*dimf])
		self.frame_z = np.reshape(self.frame_z,[dimf*dimf*dimf])
		self.frame_coords = (self.frame_coords+0.5)/dimf-0.5
		self.frame_coords = np.reshape(self.frame_coords,[dimf*dimf*dimf,3])

		self.sampling_threshold = 0.5 #final marching cubes threshold
	
		self.build_model()

	def build_model(self):
		self.vox3d = tf.placeholder(shape=[None,self.input_size,self.input_size,self.input_size,1], dtype=tf.float32, name="vox3d")
		self.point_coord = tf.placeholder(shape=[None,None,3], dtype=tf.float32, name="point_coord")
		self.point_value = tf.placeholder(shape=[None,None,1], dtype=tf.float32, name="point_value")
		self.test_z = tf.placeholder(shape=[1,self.p_dim*10], dtype=tf.float32, name="test_z")

		self.generator = self.generator_bsp_fc_qs_deeper

		self.E = self.encoder(self.vox3d, phase_train=True, reuse=False)
		self.G = self.generator(self.point_coord, self.E, phase_train=True, reuse=False)

		self.sE = self.encoder(self.vox3d, phase_train=False, reuse=True)
		self.zG = self.generator(self.point_coord, self.test_z, phase_train=False, reuse=True)

		self.loss_sp = tf.reduce_mean(tf.square(self.point_value - self.G))
		self.loss = self.loss_sp

		self.saver = tf.train.Saver(max_to_keep=2)


	def generator_bsp_fc_qs(self, points, planes, phase_train=True, reuse=False):
		with tf.variable_scope("simple_net") as scope:
			if reuse:
				scope.reuse_variables()

			shape_batch_size = tf.shape(points)[0]
			point_batch_size = tf.shape(points)[1]
			point_ones = tf.ones(shape=[shape_batch_size,point_batch_size,1], dtype=tf.float32)
			point_xy = points[:,:,0:1] * points[:,:,1:2]
			point_yz = points[:,:,1:2] * points[:,:,2:3]
			point_zx = points[:,:,2:3] * points[:,:,0:1]
			pointsx = tf.concat([tf.square(points),point_xy,point_yz,point_zx,points,point_ones],2)
			planes = tf.reshape(planes,[-1,10,self.p_dim])
			
			h1 = lrelu(tf.matmul(pointsx, planes))
			h2 = lrelu(linear(h1, self.c_dim*2, 'h2_lin'))
			h3 = lrelu(linear(h2, self.c_dim, 'h3_lin'))
			h4 = linear(h3, 1, 'h4_lin')
			h4 = tf.maximum(tf.minimum(h4, h4*0.01+0.99), h4*0.01)
			
			return h4


	def generator_bsp_fc_qs_deeper(self, points, planes, phase_train=True, reuse=False):
		with tf.variable_scope("simple_net") as scope:
			if reuse:
				scope.reuse_variables()

			shape_batch_size = tf.shape(points)[0]
			point_batch_size = tf.shape(points)[1]
			point_ones = tf.ones(shape=[shape_batch_size,point_batch_size,1], dtype=tf.float32)
			point_xy = points[:,:,0:1] * points[:,:,1:2]
			point_yz = points[:,:,1:2] * points[:,:,2:3]
			point_zx = points[:,:,2:3] * points[:,:,0:1]
			pointsx = tf.concat([tf.square(points),point_xy,point_yz,point_zx,points,point_ones],2)
			planes = tf.reshape(planes,[-1,10,self.p_dim])
			
			h1 = lrelu(tf.matmul(pointsx, planes))
			h2 = lrelu(linear(h1, self.c_dim, 'h2_lin'))
			h3 = lrelu(linear(h2, self.c_dim, 'h3_lin'))
			h4 = lrelu(linear(h3, self.c_dim, 'h4_lin'))
			h5 = linear(h4, 1, 'h5_lin')
			h5 = tf.maximum(tf.minimum(h5, h5*0.01+0.99), h5*0.01)
			
			return h5


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

			#return d_5

			#MLP below

			l4 = linear(d_5, self.p_dim*10, scope='linear_4')

			return l4
	
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
			if epoch%20==-19:
				self.save(config.checkpoint_dir, epoch)

				#test
				model_z = self.sess.run(self.sE,
					feed_dict={
						self.vox3d: self.data_voxels[batch_index_list[-1:]],
					})
				model_float = self.z2voxel(model_z)
				vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
				vertices = (vertices-0.5)/self.real_size-0.5
				#output ply sum
				write_ply_triangle(config.sample_dir+"/"+"train_"+str(self.sample_vox_size)+"_"+str(epoch)+".ply", vertices, triangles)
				print("[sample]")

		self.save(config.checkpoint_dir, self.sample_vox_size)
		fout = open("training_time_"+str(self.sample_vox_size)+".txt", 'w')
		fout.write(str(time.time() - start_time))
		fout.close()

	def test(self, config):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return

		data_hdf5_name = self.data_dir+'/'+self.dataset_load+'.hdf5'
		if os.path.exists(data_hdf5_name):
			data_dict = h5py.File(data_hdf5_name, 'r')
			self.data_voxels = data_dict['voxels'][config.start:min(len(data_dict['voxels']),config.end)]
		else:
			print("error: cannot load "+data_hdf5_name)
			exit(0)


		for t in range(len(self.data_voxels)):
			print(t,config.start,config.end)

			model_z = self.sess.run(self.sE,
				feed_dict={
					self.vox3d: self.data_voxels[t:t+1],
				})
			model_float = self.z2voxel(model_z)

			vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
			vertices = (vertices-0.5)/self.real_size-0.5

			write_obj_triangle(config.sample_dir+"/"+str(t+config.start)+".obj", vertices, triangles)

			#sample surface points
			sampled_points_normals = sample_points(vertices, triangles, 4096)
			write_ply_point_normal(config.sample_dir+"/"+str(t+config.start)+"_pc.ply", sampled_points_normals)


	def z2voxel(self, z):
		model_float = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2],np.float32)
		dimc = self.cell_grid_size
		dimf = self.frame_grid_size
		
		frame_flag = np.zeros([dimf+2,dimf+2,dimf+2],np.uint8)
		queue = []
		
		frame_batch_num = int(dimf**3/self.test_point_num)
		assert frame_batch_num>0
		
		#get frame grid values
		for i in range(frame_batch_num):
			model_out = self.sess.run(self.zG,
				feed_dict={
					self.test_z: z,
					self.point_coord: np.expand_dims(self.frame_coords[i*self.test_point_num:(i+1)*self.test_point_num], axis=0),
				})
			x_coords = self.frame_x[i*self.test_point_num:(i+1)*self.test_point_num]
			y_coords = self.frame_y[i*self.test_point_num:(i+1)*self.test_point_num]
			z_coords = self.frame_z[i*self.test_point_num:(i+1)*self.test_point_num]
			frame_flag[x_coords+1,y_coords+1,z_coords+1] = np.reshape((model_out>self.sampling_threshold).astype(np.uint8), [self.test_point_num])
		
		#get queue and fill up ones
		for i in range(1,dimf+1):
			for j in range(1,dimf+1):
				for k in range(1,dimf+1):
					maxv = np.max(frame_flag[i-1:i+2,j-1:j+2,k-1:k+2])
					minv = np.min(frame_flag[i-1:i+2,j-1:j+2,k-1:k+2])
					if maxv!=minv:
						queue.append((i,j,k))
					elif maxv==1:
						x_coords = self.cell_x+(i-1)*dimc
						y_coords = self.cell_y+(j-1)*dimc
						z_coords = self.cell_z+(k-1)*dimc
						model_float[x_coords+1,y_coords+1,z_coords+1] = 1.0
		
		print("running queue:",len(queue))
		cell_batch_size = dimc**3
		cell_batch_num = int(self.test_point_num/cell_batch_size)
		assert cell_batch_num>0
		#run queue
		while len(queue)>0:
			batch_num = min(len(queue),cell_batch_num)
			point_list = []
			cell_coords = []
			for i in range(batch_num):
				point = queue.pop(0)
				point_list.append(point)
				cell_coords.append(self.cell_coords[point[0]-1,point[1]-1,point[2]-1])
			cell_coords = np.concatenate(cell_coords, axis=0)
			model_out_batch = self.sess.run(self.zG,
				feed_dict={
					self.test_z: z,
					self.point_coord: np.expand_dims(cell_coords, axis=0),
				})
			for i in range(batch_num):
				point = point_list[i]
				model_out = model_out_batch[0,i*cell_batch_size:(i+1)*cell_batch_size,0]
				x_coords = self.cell_x+(point[0]-1)*dimc
				y_coords = self.cell_y+(point[1]-1)*dimc
				z_coords = self.cell_z+(point[2]-1)*dimc
				model_float[x_coords+1,y_coords+1,z_coords+1] = model_out
				
				if np.max(model_out)>self.sampling_threshold:
					for i in range(-1,2):
						pi = point[0]+i
						if pi<=0 or pi>dimf: continue
						for j in range(-1,2):
							pj = point[1]+j
							if pj<=0 or pj>dimf: continue
							for k in range(-1,2):
								pk = point[2]+k
								if pk<=0 or pk>dimf: continue
								if (frame_flag[pi,pj,pk] == 0):
									frame_flag[pi,pj,pk] = 1
									queue.append((pi,pj,pk))
		return model_float


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
