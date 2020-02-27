import numpy as np 
import tensorflow as tf

def lrelu(x, leak=0.01):
	return tf.maximum(x, leak*x)

def batch_norm(input, phase_train):
	return tf.contrib.layers.batch_norm(input, decay=0.99, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_train)

def instance_norm(input, phase_train):
	return tf.contrib.layers.instance_norm(input)

def linear(input_, output_size, scope, add_reg=False):
	shape = input_.get_shape().as_list()
	with tf.variable_scope(scope):
		matrix = tf.get_variable("Matrix", [shape[1], output_size], initializer=tf.contrib.layers.xavier_initializer())#tf.random_normal_initializer(stddev=0.02)
		bias = tf.get_variable("bias", [output_size], initializer=tf.zeros_initializer())
		if add_reg:
			tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, matrix)
		print("linear","in",shape,"out",(shape[0],output_size))
		return tf.matmul(input_, matrix) + bias

def conv2d(input_, shape, strides, scope, padding="SAME", add_reg=False):
	with tf.variable_scope(scope):
		matrix = tf.get_variable('Matrix', shape, initializer=tf.contrib.layers.xavier_initializer())#tf.truncated_normal_initializer(stddev=0.02)
		bias = tf.get_variable('bias', [shape[-1]], initializer=tf.zeros_initializer())
		if add_reg:
			tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, matrix)
		conv = tf.nn.conv2d(input_, matrix, strides=strides, padding=padding)
		conv = tf.nn.bias_add(conv, bias)
		print("conv2d","in",input_.shape,"out",conv.shape)
		return conv

def conv2d_nobias(input_, shape, strides, scope, padding="SAME", add_reg=False):
	with tf.variable_scope(scope):
		matrix = tf.get_variable('Matrix', shape, initializer=tf.contrib.layers.xavier_initializer())#tf.truncated_normal_initializer(stddev=0.02)
		if add_reg:
			tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, matrix)
		conv = tf.nn.conv2d(input_, matrix, strides=strides, padding=padding)
		print("conv2d","in",input_.shape,"out",conv.shape)
		return conv

def conv3d(input_, shape, strides, scope, padding="SAME"):
	with tf.variable_scope(scope):
		matrix = tf.get_variable("Matrix", shape, initializer=tf.contrib.layers.xavier_initializer())
		bias = tf.get_variable("bias", [shape[-1]], initializer=tf.zeros_initializer())
		conv = tf.nn.conv3d(input_, matrix, strides=strides, padding=padding)
		conv = tf.nn.bias_add(conv, bias)
		print("conv3d","in",input_.shape,"out",conv.shape)
		return conv

def deconv3d(input_, shape, out_shape, strides, scope, padding="SAME"):
	with tf.variable_scope(scope):
		matrix = tf.get_variable("Matrix", shape, initializer=tf.contrib.layers.xavier_initializer())
		bias = tf.get_variable("bias", [shape[-2]], initializer=tf.zeros_initializer())
		conv = tf.nn.conv3d_transpose(input_, matrix, out_shape, strides=strides, padding=padding)
		conv = tf.nn.bias_add(conv, bias)
		print("deconv3d","in",input_.shape,"out",conv.shape)
		return conv


