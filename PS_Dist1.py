import argparse
import sys
import tensorflow as tf
FLAGS = None

# Parameters
lr = 0.001
epoches = 15
batch_size = 100
display_step = 1


def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

  print(" ---------------------------------- ")
  print("job name = %s"   % FLAGS.job_name)
  print("task index = %d" % FLAGS.task_index)
  print(" ---------------------------------- ")
  print("ps  = %s"   % ps_hosts)
  print("worker = %s" % worker_hosts)
  print(" ---------------------------------- ")

  if FLAGS.job_name is None or FLAGS.job_name =="":
  	raise ValueError("Must specify a job_name")
  if FLAGS.task_index is None or FLAGS.task_index =="":
  	raise ValueError("Must specify a task_index")

  if FLAGS.job_name == "ps":
  	server.join()
  elif FLAGS.job_name == "worker":

  	with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):
  		X = tf.placeholder(dtype=tf.float32,shape=[None, 784])
  		Y = tf.placeholder(dtype=tf.float32,shape=[None, 10])
  		# Store layers weight & bias
  		
  		h1 = tf.Variable(tf.random_normal([784, 256]))
  		b1 = tf.Variable(tf.random_normal([256]))

  		h2 = tf.Variable(tf.random_normal([256, 128]))
  		b2 = tf.Variable(tf.random_normal([128]))

  		h3 = tf.Variable(tf.random_normal([128, 10]))
  		b3 = tf.Variable(tf.random_normal([10]))
  		

  		# Hidden fully connected layer with 256 neurons
  		layer_1 = tf.add(tf.matmul(X, h1), b1)
  		Act_1 = tf.sigmoid(layer_1)
  		# Hidden fully connected layer with 256 neurons
  		layer_2 = tf.add(tf.matmul(Act_1, h2), b2)
  		Act_2 = tf.sigmoid(layer_2)

  		#Output fully connected layer with a neuron for each class
  		logits = tf.add(tf.matmul(Act_2, h3), b3)

  		loss_op = tf.losses.softmax_cross_entropy(Y,logits)

  		global_step = tf.train.get_or_create_global_step()

  		train_op = tf.train.AdagradOptimizer(0.01).minimize(loss_op, global_step=global_step)

  		hooks=[tf.train.StopAtStepHook(last_step=1000000)]

  		with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0), checkpoint_dir="/tmp/train_logs", hooks=hooks) as mon_sess:
  			while not mon_sess.should_stop():
  				mon_sess.run(train_op)
  				
  		'''
  		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
  			sess.run(init)
  			# Training cycle
  			for epoch in range(training_epochs):
  				avg_cost = 0.
  				total_batch = int(mnist.train.num_examples/batch_size)
  				# Loop over all batches
  				for i in range(total_batch):
  					batch_x, batch_y = mnist.train.next_batch(batch_size)
  					# Run optimization op (backprop) and cost op (to get loss value)
  					_, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
  					# Compute average loss
  					avg_cost += c / total_batch
  					# Display logs per epoch step
  					if epoch % display_step == 0:
  						print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
  					print("Optimization Finished!")

  				# Test model
  				pred = tf.nn.softmax(logits)  # Apply softmax to logits
  				correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
  				# Calculate accuracy
  				accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  				print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
  		'''
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)