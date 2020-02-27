import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np

from modelAE import BSP_AE
from modelSVR import BSP_SVR

import tensorflow as tf
import h5py

flags = tf.app.flags
flags.DEFINE_integer("phase", 1, "phase 0 = continuous, phase 1 = hard discrete, phase 2 = hard discrete with L_overlap, phase 3 = soft discrete [1]")
#phase 0 continuous for better convergence
#phase 1 hard discrete for bsp
#phase 2 hard discrete for bsp with L_overlap
#phase 3 soft discrete for bsp
#use [phase 0 -> phase 1] or [phase 0 -> phase 2] or [phase 0 -> phase 3]
flags.DEFINE_integer("epoch", 0, "Epoch to train [0]")
flags.DEFINE_integer("iteration", 0, "Iteration to train. Either epoch or iteration need to be zero [0]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate for adam [0.00002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_string("dataset", "all_vox256_img", "The name of dataset")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data/all_vox256_img/", "Root directory of dataset [data]")
flags.DEFINE_string("sample_dir", "./samples/", "Directory name to save the image samples [samples]")
flags.DEFINE_integer("sample_vox_size", 64, "Voxel resolution for coarse-to-fine training [64]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_integer("start", 0, "In testing, output shapes [start:end]")
flags.DEFINE_integer("end", 16, "In testing, output shapes [start:end]")
flags.DEFINE_boolean("ae", False, "True for ae [False]")
flags.DEFINE_boolean("svr", False, "True for svr [False]")
flags.DEFINE_boolean("getz", False, "True for getting latent codes [False]")
FLAGS = flags.FLAGS

def main(_):
	if not os.path.exists(FLAGS.sample_dir):
		os.makedirs(FLAGS.sample_dir)

	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
	#run_config = tf.ConfigProto(gpu_options=gpu_options)
	run_config = tf.ConfigProto()
	run_config.gpu_options.allow_growth=True

	if FLAGS.ae:
		with tf.Session(config=run_config) as sess:
			bsp_ae = BSP_AE(
					sess,
					FLAGS.phase,
					FLAGS.sample_vox_size,
					is_training = FLAGS.train or FLAGS.getz,
					dataset_name=FLAGS.dataset,
					checkpoint_dir=FLAGS.checkpoint_dir,
					sample_dir=FLAGS.sample_dir,
					data_dir=FLAGS.data_dir)

			if FLAGS.train:
				bsp_ae.train(FLAGS)
			elif FLAGS.getz:
				bsp_ae.get_z(FLAGS)
			else:
				if FLAGS.phase==0:
					bsp_ae.test_dae3(FLAGS)
				else:
					#bsp_ae.test_bsp(FLAGS)
					bsp_ae.test_mesh_point(FLAGS)
					#bsp_ae.test_mesh_obj_material(FLAGS)
	elif FLAGS.svr:
		with tf.Session(config=run_config) as sess:
			bsp_svr = BSP_SVR(
					sess,
					FLAGS.phase,
					FLAGS.sample_vox_size,
					is_training = FLAGS.train,
					dataset_name=FLAGS.dataset,
					checkpoint_dir=FLAGS.checkpoint_dir,
					sample_dir=FLAGS.sample_dir,
					data_dir=FLAGS.data_dir)

			if FLAGS.train:
				bsp_svr.train(FLAGS)
			else:
				#bsp_svr.test_bsp(FLAGS)
				bsp_svr.test_mesh_point(FLAGS)
				#bsp_svr.test_mesh_obj_material(FLAGS)
	else:
		print("Please specify an operation: ae or svr?")

if __name__ == '__main__':
	tf.app.run()
