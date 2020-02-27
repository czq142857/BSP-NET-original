import os
import time
import math
import random
import tensorflow as tf
import numpy as np
import h5py
import cv2

from bspt_2d import *
from ops import *

class IMSEG(object):
	def __init__(self, sess, phase, sample_vox_size, is_training = False, ef_dim=32, gf_dim=64, p_dim=256, dataset_name='default', checkpoint_dir=None, sample_dir=None, data_dir='./data'):
		"""
		Args:
			too lazy to explain
		"""
		self.sess = sess
		self.phase = phase
		#phase 0 continuous for better convergence
		#phase 1,2 discrete for bsp

		self.sample_vox_size = sample_vox_size
		self.point_batch_size = sample_vox_size*sample_vox_size
		self.shape_batch_size = 20

		self.p_dim = p_dim
		self.ef_dim = ef_dim
		self.gf_dim = gf_dim

		self.dataset_name = dataset_name
		self.checkpoint_dir = checkpoint_dir
		self.data_dir = data_dir
		
		data_hdf5_name = self.data_dir+'/'+self.dataset_name+'.hdf5'
		if os.path.exists(data_hdf5_name):
			self.data_dict = h5py.File(data_hdf5_name, 'r')
			self.data_voxels = self.data_dict['pixels'][:]
		else:
			print("error: cannot load "+data_hdf5_name)
			exit(0)
		
		self.build_model(phase)

	def build_model(self, phase):
		#get coords
		dim = self.sample_vox_size
		self.coords = np.zeros([dim,dim,2],np.float32)
		for i in range(dim):
			for j in range(dim):
				self.coords[i,j,0] = i
				self.coords[i,j,1] = j
		self.coords = (self.coords+0.5)/dim-0.5
		self.coords = np.tile(np.reshape(self.coords,[1,self.point_batch_size,2]),[self.shape_batch_size,1,1])

		self.vox3d = tf.placeholder(shape=[self.shape_batch_size,self.sample_vox_size,self.sample_vox_size,1], dtype=tf.float32, name="vox3d")
		self.point_coord = tf.constant(self.coords, dtype=tf.float32, name="point_coord")
		self.point_value = tf.reshape(self.vox3d, [self.shape_batch_size,self.point_batch_size,1])
		self.plane_m = tf.placeholder(shape=[self.shape_batch_size,2,self.p_dim], dtype=tf.float32, name="plane_m")
		self.plane_b = tf.placeholder(shape=[self.shape_batch_size,1,self.p_dim], dtype=tf.float32, name="plane_b")

		if phase==0:
			self.generator = self.generator0
		elif phase==1 or phase==2:
			self.generator = self.generator1

		self.E_m, self.E_b = self.encoder(self.vox3d, phase_train=True, reuse=False)
		self.G, _, self.G2, self.cw2, self.cw3 = self.generator(self.point_coord, self.E_m, self.E_b, phase_train=True, reuse=False)

		self.sE_m, self.sE_b = self.encoder(self.vox3d, phase_train=False, reuse=True)
		self.sG, self.sG_max, self.sG2, _, _ = self.generator(self.point_coord, self.sE_m, self.sE_b, phase_train=False, reuse=True)
		self.zG, self.zG_max, self.zG2, _, _ = self.generator(self.point_coord, self.plane_m, self.plane_b, phase_train=False, reuse=True)

		if phase==0:
			self.loss_sp = tf.reduce_mean(tf.square(self.point_value - self.G))
			self.loss = self.loss_sp + tf.reduce_sum(tf.abs(self.cw3-1))*1.0 + (tf.reduce_sum(tf.maximum(self.cw2-1,0)) - tf.reduce_sum(tf.minimum(self.cw2,0)))*1.0
		elif phase==1:
			self.loss_sp = tf.reduce_mean((1-self.point_value)*tf.abs(tf.minimum(self.G,1)-1) + self.point_value*tf.abs(tf.maximum(self.G,0)))
			self.loss = self.loss_sp
		elif phase==2:
			self.loss_sp = tf.reduce_mean((1-self.point_value)*tf.abs(tf.minimum(self.G,1)-1) + self.point_value*tf.abs(tf.maximum(self.G,0)))
			self.bmask = tf.cast(self.G2<0.01, tf.float32) * tf.cast( tf.reduce_sum(tf.cast(self.G2<0.01, tf.float32),axis=2, keepdims=True)>1, tf.float32)
			self.loss = self.loss_sp - tf.reduce_mean(self.G2*self.point_value*self.bmask)
			
		self.saver = tf.train.Saver(max_to_keep=10)
		
		
	def generator0(self, points, plane_m, plane_b, phase_train=True, reuse=False):
		with tf.variable_scope("simple_net") as scope:
			if reuse:
				scope.reuse_variables()
			
			#level 1
			h1 = tf.matmul(points, plane_m) + plane_b
			h1 = tf.maximum(h1, 0)
			
			#level 2
			convex_layer_weights = tf.get_variable("convex_layer_weights", [self.p_dim, self.gf_dim], initializer=tf.random_normal_initializer(stddev=0.02))
			h2 = tf.matmul(h1, convex_layer_weights)
			h2 = tf.maximum(tf.minimum(1-h2, 1), 0)
			
			#level 3
			concave_layer_weights = tf.get_variable("concave_layer_weights", [self.gf_dim, 1], initializer=tf.random_normal_initializer(stddev=0.02))
			h3 = tf.matmul(h2, concave_layer_weights)
			h3 = tf.maximum(tf.minimum(h3, 1), 0)
			h3_max = tf.reduce_max(h2, axis=2, keepdims=True)
			
		return h3, h3_max, h2, convex_layer_weights, concave_layer_weights

	def generator1(self, points, plane_m, plane_b, phase_train=True, reuse=False):
		with tf.variable_scope("simple_net") as scope:
			if reuse:
				scope.reuse_variables()
			
			#level 1
			h1 = tf.matmul(points, plane_m) + plane_b
			h1 = tf.maximum(h1, 0)
			
			#level 2
			convex_layer_weights = tf.get_variable("convex_layer_weights", [self.p_dim, self.gf_dim], initializer=tf.random_normal_initializer(stddev=0.02))
			convex_layer_weights = tf.cast(convex_layer_weights>0.01, convex_layer_weights.dtype)
			h2 = tf.matmul(h1, convex_layer_weights)
			
			#level 3
			h3 = tf.reduce_min(h2, axis=2, keepdims=True)
			h3_01 = tf.maximum(tf.minimum(1-tf.stop_gradient(h3), 1), 0)

		return h3, h3_01, h2, convex_layer_weights, None

	def encoder(self, inputs, phase_train=True, reuse=False):
		with tf.variable_scope("encoder") as scope:
			if reuse:
				scope.reuse_variables()
			
			d_1 = conv2d(inputs, shape=[4, 4, 1, self.ef_dim], strides=[1,2,2,1], scope='conv_1')
			d_1 = lrelu(d_1)

			d_2 = conv2d(d_1, shape=[4, 4, self.ef_dim, self.ef_dim*2], strides=[1,2,2,1], scope='conv_2')
			d_2 = lrelu(d_2)
			
			d_3 = conv2d(d_2, shape=[4, 4, self.ef_dim*2, self.ef_dim*4], strides=[1,2,2,1], scope='conv_3')
			d_3 = lrelu(d_3)

			d_4 = conv2d(d_3, shape=[4, 4, self.ef_dim*4, self.ef_dim*8], strides=[1,2,2,1], scope='conv_4')
			d_4 = lrelu(d_4)

			d_5 = conv2d(d_4, shape=[4, 4, self.ef_dim*8, self.ef_dim*8], strides=[1,1,1,1], scope='conv_5', padding="VALID")
			d_5 = lrelu(d_5)
			d_5 = tf.reshape(d_5,[-1, self.ef_dim*8])

			l1 = linear(d_5, self.ef_dim*16, scope='linear_1')
			l1 = lrelu(l1)

			l2 = linear(l1, self.ef_dim*32, scope='linear_2')
			l2 = lrelu(l2)

			l3 = linear(l2, self.ef_dim*64, scope='linear_3')
			l3 = lrelu(l3)

			l4_m = linear(l3, self.p_dim*2, scope='linear_4m')
			l4_b = linear(l3, self.p_dim, scope='linear_4b')

			l4_m = tf.reshape(l4_m,[-1, 2, self.p_dim])
			l4_b = tf.reshape(l4_b,[-1, 1, self.p_dim])

			return l4_m, l4_b
	
	def train(self, config):
		ae_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss)
		self.sess.run(tf.global_variables_initializer())
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			
		shane_num = len(self.data_voxels)
		batch_index_list = np.arange(shane_num)
		
		print("\n\n----------net summary----------")
		print("training samples   ", shane_num)
		print("-------------------------------\n\n")
		
		counter = 0
		start_time = time.time()
		assert config.epoch==0 or config.iteration==0
		training_epoch = config.epoch + int(config.iteration/shane_num)
		
		batch_num = int(shane_num/self.shape_batch_size)
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
					})
				avg_loss_sp += errSP
				avg_loss_tt += errTT
				avg_num += 1
			print(str(self.sample_vox_size)+" Epoch: [%2d/%2d] time: %4.4f, loss_sp: %.8f, loss_total: %.8f" % (epoch, training_epoch, time.time() - start_time, avg_loss_sp/avg_num, avg_loss_tt/avg_num))
			if epoch%100==99:
				self.test_1(config)
				
		if config.phase==0:
			self.save(config.checkpoint_dir, self.sample_vox_size)
		else:
			self.save(config.checkpoint_dir, training_epoch)

	def test_1(self, config):
		if config.phase==0:
			outG = self.sG
		elif config.phase==1 or config.phase==2:
			outG = self.sG_max
		t = 0
		batch_voxels = self.data_voxels[t:t+self.shape_batch_size]
		model_out = self.sess.run(outG,
			feed_dict={
				self.vox3d: batch_voxels,
			})
		imgs = np.clip(np.resize(model_out,[self.shape_batch_size,self.sample_vox_size,self.sample_vox_size])*256, 0, 255).astype(np.uint8)
		for t in range(self.shape_batch_size):
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_out.png", imgs[t])
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_gt.png", batch_voxels[t]*255)
		
		if config.phase==1 or config.phase==2:
			image_out_size = 256
			w2 = self.sess.run(self.cw2, feed_dict={})

			start_n = config.start
			batch_voxels = self.data_voxels[start_n:start_n+self.shape_batch_size]
			model_out, out_m, out_b = self.sess.run([self.sG2, self.sE_m, self.sE_b],
				feed_dict={
					self.vox3d: batch_voxels,
				})
			model_out = np.resize(model_out,[self.shape_batch_size,self.sample_vox_size,self.sample_vox_size,self.gf_dim])

			for t in range(self.shape_batch_size):
				bsp_convex_list = []
				color_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
				color_idx_list = []

				for i in range(self.gf_dim):
					min_v = np.min(model_out[t,:,:,i])
					if min_v<0.01:
						box = []
						for j in range(self.p_dim):
							if w2[j,i]>0.01:
								a = -out_m[t,0,j]
								b = -out_m[t,1,j]
								d = -out_b[t,0,j]
								box.append([a,b,d])
						if len(box)>0:
							bsp_convex_list.append(np.array(box,np.float32))
							color_idx_list.append(i%len(color_list))

				#print(bsp_convex_list)
				print(len(bsp_convex_list))
				
				#convert bspt to mesh
				vertices = []
				polygons = []
				polygons_color = []

				img_out = np.full([image_out_size,image_out_size,3],255,np.uint8)
				for i in range(len(bsp_convex_list)):
					vg, tg = digest_bsp(bsp_convex_list[i], bias=0)
					cg = color_list[color_idx_list[i]]
					for j in range(len(tg)):
						x1 = ((vg[tg[j][0]][1]+0.5)*image_out_size).astype(np.int32)
						y1 = ((vg[tg[j][0]][0]+0.5)*image_out_size).astype(np.int32)
						x2 = ((vg[tg[j][1]][1]+0.5)*image_out_size).astype(np.int32)
						y2 = ((vg[tg[j][1]][0]+0.5)*image_out_size).astype(np.int32)
						cv2.line(img_out, (x1,y1), (x2,y2), cg, thickness=1)
				
				cv2.imwrite(config.sample_dir+"/"+str(t)+"_bsp.png", img_out)

		print("[sample]")

	#output bsp shape with color
	def test_bsp(self, config):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		image_out_size = 256
		w2 = self.sess.run(self.cw2, feed_dict={})

		start_n = config.start
		batch_voxels = self.data_voxels[start_n:start_n+self.shape_batch_size]
		model_out, out_m, out_b = self.sess.run([self.sG2, self.sE_m, self.sE_b],
			feed_dict={
				self.vox3d: batch_voxels,
			})
		model_out = np.resize(model_out,[self.shape_batch_size,self.sample_vox_size,self.sample_vox_size,self.gf_dim])

		for t in range(self.shape_batch_size):
			bsp_convex_list = []
			color_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
			color_idx_list = []

			for i in range(self.gf_dim):
				min_v = np.min(model_out[t,:,:,i])
				if min_v<0.01:
					box = []
					for j in range(self.p_dim):
						if w2[j,i]>0.01:
							a = -out_m[t,0,j]
							b = -out_m[t,1,j]
							d = -out_b[t,0,j]
							box.append([a,b,d])
					if len(box)>0:
						bsp_convex_list.append(np.array(box,np.float32))
						color_idx_list.append(i%len(color_list))

			#print(bsp_convex_list)
			print(len(bsp_convex_list))
			
			#convert bspt to mesh
			vertices = []
			polygons = []
			polygons_color = []

			img_out = np.full([image_out_size,image_out_size,3],255,np.uint8)
			for i in range(len(bsp_convex_list)):
				vg, tg = digest_bsp(bsp_convex_list[i], bias=0)
				cg = color_list[color_idx_list[i]]
				for j in range(len(tg)):
					x1 = ((vg[tg[j][0]][1]+0.5)*image_out_size).astype(np.int32)
					y1 = ((vg[tg[j][0]][0]+0.5)*image_out_size).astype(np.int32)
					x2 = ((vg[tg[j][1]][1]+0.5)*image_out_size).astype(np.int32)
					y2 = ((vg[tg[j][1]][0]+0.5)*image_out_size).astype(np.int32)
					cv2.line(img_out, (x1,y1), (x2,y2), cg, thickness=1)
			
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_bsp.png", img_out)

	#output h3
	def test_dae3(self, config):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		t = 0
		batch_voxels = self.data_voxels[t:t+self.shape_batch_size]
		model_out = self.sess.run(self.sG,
			feed_dict={
				self.vox3d: batch_voxels,
			})
		imgs = np.clip(np.resize(model_out,[self.shape_batch_size,self.sample_vox_size,self.sample_vox_size])*256, 0, 255).astype(np.uint8)
		for t in range(self.shape_batch_size):
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_out.png", imgs[t])
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_gt.png", batch_voxels[t]*255)
		print("[sample]")
	
	@property
	def model_dir(self):
		return "{}_{}".format(
				self.dataset_name, self.sample_vox_size)
			
	def save(self, checkpoint_dir, step):
		model_name = "IMSEG.model"
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
