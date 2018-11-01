from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf 
import math
import sys
import time
import tempfile
import numpy as np
import argparse

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


FLAGS = None
IMAGE_PIXELS = 28
batch_size = 100
lr = 0.01
cluster = None
server = None
is_chief = False
n_iterations = 1000

def main(_):

	# Creates an Dataset for MNIST Data.

	(train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.mnist.load_data()
	
	train_labels = train_labels[:1000]
	test_labels  =  test_labels[:1000]

	train_images = train_images[:1000].reshape(-1,28*28) / 255.0
	test_images  = test_images[:1000].reshape(-1,28*28) / 255.0
	
	print(" --------------Dataset-------------------- ")

	print(train_images, train_labels)
	print(test_images, test_labels)
	

	print(" --------------End Dataset -------------------- ")

	ps_spec = FLAGS.ps_hosts.split(",")
	worker_spec = FLAGS.worker_hosts.split(",")
	print(" ---------------------------------- ")
	print("job name = %s"   % FLAGS.job_name)
	print("task index = %d" % FLAGS.task_index)
	print(" ---------------------------------- ")
	print("ps  = %s"   % ps_spec)
	print("worker = %s" % worker_spec)
	print(" ---------------------------------- ")

	if FLAGS.job_name is None or FLAGS.job_name =="":
		raise ValueError("Must specify a job_name")
	if FLAGS.task_index is None or FLAGS.task_index =="":
		raise ValueError("Must specify a task_index")


	# Cluster Specification 
	cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker':worker_spec})

	# start server
	server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
	

	# ------------ Between-graph replication model -----------------------

	if FLAGS.job_name =='ps':
		server.join()
	elif FLAGS.job_name =='worker':
		with tf.device(tf.train.replica_device_setter(worker_device= "/job:worker/task:%d" 
			% FLAGS.task_index,cluster=cluster)):

			# defining training graph 

			# defining input 
			x_ = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="x-input")

			y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="y-input")

			# from input to first hidden layer
			w1 = tf.Variable(tf.truncated_normal(shape=[784, 128],stddev=1.0/ math.sqrt(float(784)),dtype=tf.float32),name='w1')
			b1 = tf.Variable(tf.zeros([128]),name='b1')
			hid_1 = tf.add(tf.matmul(x_,w1),b1)
			Act_1 = tf.sigmoid(hid_1)
			print(" w1 --------------------------------------- ")
			# from first hidden layer to second
			w2 = tf.Variable(tf.truncated_normal(shape=[128, 64],stddev=1.0/ math.sqrt(float(128)),dtype=tf.float32),name='w2')
			b2 = tf.Variable(tf.zeros([64]),name='b2')
			hid_2 = tf.add(tf.matmul(Act_1,w2),b2)
			Act_2 = tf.sigmoid(hid_2)
			print(" w2 --------------------------------------- ")

			# from second to output
			w3 = tf.Variable(tf.truncated_normal(shape=[64,10],stddev=1.0/ math.sqrt(float(64)),dtype=tf.float32),name='w3')
			b3 = tf.Variable(tf.zeros([10]),name='b3')
			
			logits = tf.add(tf.matmul(Act_2,w3),b3)

			# build loss (softmax cross entropy)
			
			loss = tf.losses.softmax_cross_entropy(y_,logits)

			print(" w3 --------------------------------------- ")

			# define optimization
			global_step = tf.train.get_or_create_global_step()
			optimizer = tf.train.GradientDescentOptimizer(lr)
			training_op = optimizer.minimize(loss, global_step=global_step)

			print(" y --------------------------------------- ")
			
			# use accuracy for evaluation
    		
    		#predictions = tf.nn.softmax(logits, axis=1)
    		#eval_metric_ops ={
    		#	'accuracy': tf.metrics.accuracy(labels, predictions, name='accuracy')
    		#}

			if FLAGS.sync_replicas:
				replicas_to_aggregate = len(worker_spec)
				print(" w1 -------------FLAGS.sync_replicas----------------- ")
				optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=replicas_to_aggregate,
				total_num_replicas=len(worker_spec), name="mnist_synch_replicas")

			#y_hat = tf.round(tf.argmax(tf.nn.softmax(logits), 1))
			#y_hat = tf.cast(y_hat, tf.uint8)
			#y_hat = tf.reshape(y_hat, [-1, 1])

			print(" w1 -------------optimizer----------------- ")

			sync_replicas_hook = optimizer.make_session_run_hook(FLAGS.task_index == 0)
			hooks=[tf.train.StopAtStepHook(last_step=1000000)]

			#---------------training session--------------

			with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0), 
				checkpoint_dir="/tmp/train_logs", hooks=hooks) as mon_sess:
				while not mon_sess.should_stop():
					print("loss", loss )
					mon_sess.run(training_op, loss)

			print('training finished')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.register("type","bool",lambda v:v.lower()=="true")

	parser.add_argument(
		"--ps_hosts",
		type =str,
		default="",
		help="comma-separate the address"
		)

	parser.add_argument(
		"--worker_hosts",
		type =str,
		default="",
		help="comma-separate the address"
		)

	parser.add_argument(
		"--job_name",
		type =str,
		default="",
		help="ps or worker"
		)

	parser.add_argument(
		"--task_index",
		type =int,
		default=0,
		help="index=0 or 1"
		)

	parser.add_argument(
		"--sync_replicas",
		type =bool,
		default=True,
		help="sync_replicas"
		)
	
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)
