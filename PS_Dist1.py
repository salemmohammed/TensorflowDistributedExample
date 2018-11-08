import argparse
import sys
import os
import time
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
import numpy as np
from timeit import default_timer as timer

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None
batch_size = 500


def main(_):
  f = open("pids.txt", 'a')
  f.write("{} ".format(os.getpid()))
  f.flush()
  print("\n\n -------------PRINTING FLAGS----------- \n")
  print("{}".format(FLAGS))
  print("\n ---------------------------------------- \n\n")

  #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)
  
  #print("\n\n ---------------SERVER DEF------------- \n")
  #print("{}".format(server.server_def))
  #print("is chief: {}".format(is_chief))
  #print("\n ---------------------------------------- \n\n")

  is_chief = (FLAGS.task_index == 0)
  num_workers = len(worker_hosts)

  if FLAGS.job_name == "ps":
    print('--- Parameter Server Ready ---')
    server.join()

  elif FLAGS.job_name == "worker":

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # Assigns ops to the local worker by default.

    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster = cluster)):

      global_step = tf.train.get_or_create_global_step()

      with tf.name_scope('input'):
        X = tf.placeholder(dtype=tf.float32,shape=[None, 784])
        Y = tf.placeholder(dtype=tf.float32,shape=[None, 10])

      with tf.name_scope("weights"):
        W1 = tf.Variable(tf.random_normal([784, 256]))
        W2 = tf.Variable(tf.random_normal([256,64]))
        W3 = tf.Variable(tf.random_normal([64, 10]))
      with tf.name_scope("biases"):
        b1 = tf.Variable(tf.zeros([256]))
        b2 = tf.Variable(tf.zeros([64]))
        b3 = tf.Variable(tf.zeros([10]))

      with tf.name_scope("softmax"):
        z2 = tf.add(tf.matmul(X,W1),b1)
        a2 = tf.nn.sigmoid(z2)
        z3 = tf.add(tf.matmul(a2,W2),b2)
        a3 = tf.nn.sigmoid(z3)
        z4 = tf.add(tf.matmul(a3,W3),b3)
        out  = tf.nn.softmax(z4)

      # Build model...
      with tf.name_scope('cross_entropy'):
        loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(out), reduction_indices=[1]))

      
      
      
      print('{} workers defined'.format(num_workers))

      with tf.name_scope("train"):
        grad = tf.train.GradientDescentOptimizer(learning_rate=0.0014)
        begin = timer()
        syn = tf.train.SyncReplicasOptimizer(grad, replicas_to_aggregate=num_workers, total_num_replicas=num_workers)
        begin_again = timer()
        elapse = begin_again - begin
        print("elapse", elapse)
        train_op = syn.minimize(loss, global_step=global_step)
        sync_replicas_hook = syn.make_session_run_hook(is_chief)

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(out,1),tf.argmax(Y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


  stop_hook= tf.train.StopAtStepHook(last_step=10000)
  
  print(" Start Training .... ")

  with tf.train.MonitoredTrainingSession( master=server.target, is_chief=is_chief, hooks=[sync_replicas_hook]) as mon_sess:

    step = 0
    i = 0
    Average_Computation_Time = 0
    try:

      while not mon_sess.should_stop():

        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        #print(batch_xs)
        # Run a training step asynchronously.
        # See <a href="./../api_docs/python/tf/train/SyncReplicasOptimizer"><code>tf.train.SyncReplicasOptimizer</code></a> for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.

        start = timer()

        _, global_step_value, loss_value = mon_sess.run([train_op, global_step, loss],feed_dict={X: batch_xs, Y: batch_ys})

        start2 = timer()

        end = start2 - start

        i= i + 1

        Average_Computation_Time = Average_Computation_Time + end

        #print("End ------- : ", end)

        step = step + 1

        if step % 100 == 0:
          print(" Finish Step %d, Global Step %d, (Ave-Time: %.5f),  (Loss: %.2f)" % (step, global_step_value, end, loss_value))

      except tf.errors.OutOfRangeError:
        print('training finished, number of epochs reached')

    Average_Computation_Time = Average_Computation_Time / i
    print("(Average Computation Time: %.5f),  (Counter: %d)" % (Average_Computation_Time, i))

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