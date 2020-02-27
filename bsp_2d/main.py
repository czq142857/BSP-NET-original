import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np

from model import IMSEG

import tensorflow as tf
import h5py

flags = tf.app.flags
flags.DEFINE_integer("phase", 1, "phase0=continuous phase1=discrete [1]")
flags.DEFINE_integer("epoch", 0, "Epoch to train [0]")
flags.DEFINE_integer("iteration", 0, "Iteration to train. Either epoch or iteration need to be zero [0]")
flags.DEFINE_float("learning_rate", 0.00002, "Learning rate for adam [0.00002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_string("dataset", "03001627_vox", "The name of dataset")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_integer("sample_vox_size", 64, "Voxel resolution for coarse-to-fine training [64]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_integer("start", 0, "In testing, output shapes [start:start+16]")
FLAGS = flags.FLAGS

def main(_):
	if not os.path.exists(FLAGS.sample_dir):
		os.makedirs(FLAGS.sample_dir)

	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
	#run_config = tf.ConfigProto(gpu_options=gpu_options)
	run_config = tf.ConfigProto()
	run_config.gpu_options.allow_growth=True

	with tf.Session(config=run_config) as sess:
		imseg = IMSEG(
				sess,
				FLAGS.phase,
				FLAGS.sample_vox_size,
				is_training = FLAGS.train,
				dataset_name=FLAGS.dataset,
				checkpoint_dir=FLAGS.checkpoint_dir,
				sample_dir=FLAGS.sample_dir,
				data_dir=FLAGS.data_dir)

		if FLAGS.train:
			imseg.train(FLAGS)
		else:
			if FLAGS.phase==1:
				imseg.test_bsp(FLAGS)
			else:
				imseg.test_dae3(FLAGS)

if __name__ == '__main__':
	tf.app.run()
