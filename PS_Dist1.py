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
batch_size = 100


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

  #train, test = tf.keras.datasets.mnist.load_data()
  #mnist_x, mnist_y = train
  #mnist_ds = tf.data.Dataset.from_tensor_slices(mnist_x)
  
  #print("\n\n ---------------SERVER DEF------------- \n")
  #print("{}".format(server.server_def))
  #print("is chief: {}".format(is_chief))
  #print("\n ---------------------------------------- \n\n")

  is_chief = (FLAGS.task_index == 0)
  num_workers = len(worker_hosts)
  print("num_workers" , num_workers)
  if FLAGS.job_name == "ps":
    print('--- Parameter Server Ready ---')
    server.join()

  elif FLAGS.job_name == "worker":

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # Assigns ops to the local worker by default.
    
    #with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster = cluster)):

    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % (FLAGS.task_index),cluster=cluster)):

      global_step = tf.contrib.framework.get_or_create_global_step()


      with tf.name_scope('input'):
        X = tf.placeholder(dtype=tf.float32,shape=[None, 784])
        Y = tf.placeholder(dtype=tf.float32,shape=[None, 10])
        keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

      with tf.name_scope("weights"):
        W1 = tf.Variable(tf.random_normal([784, 512]),dtype=tf.float32)
        W2 = tf.Variable(tf.random_normal([512,256]),dtype=tf.float32)
        W3 = tf.Variable(tf.random_normal([256,128]),dtype=tf.float32)
        W4 = tf.Variable(tf.random_normal([128, 64]),dtype=tf.float32)
        W5 = tf.Variable(tf.random_normal([64, 10]),dtype=tf.float32)
        s1 = tf.size(W1)
        print("the size ----------------------------------- ", s1) 
      with tf.name_scope("biases"):
        b1 = tf.Variable(tf.zeros([512]),dtype=tf.float32)
        b2 = tf.Variable(tf.zeros([256]),dtype=tf.float32)
        b3 = tf.Variable(tf.zeros([128]),dtype=tf.float32)
        b4 = tf.Variable(tf.zeros([64]),dtype=tf.float32)
        b5 = tf.Variable(tf.zeros([10]),dtype=tf.float32)

      with tf.name_scope("softmax"):
        z2 = tf.add(tf.matmul(X,W1),b1)
        a2 = tf.nn.sigmoid(z2)

        z3 = tf.add(tf.matmul(a2,W2),b2)
        a3 = tf.nn.sigmoid(z3)

        z4 = tf.add(tf.matmul(a3,W3),b3)
        a4 = tf.nn.sigmoid(z4)

        z5 = tf.add(tf.matmul(a4,W4),b4)
        a5 = tf.nn.sigmoid(z5)

        z6 = tf.add(tf.matmul(a5,W5),b5)
        out  = tf.nn.softmax(z6)

      # Build model...
      with tf.name_scope('cross_entropy'):
        loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(out), reduction_indices=[1]))

      
      
      
      print('{} workers defined'.format(num_workers))

      with tf.name_scope("train"):
        grad = tf.train.GradientDescentOptimizer(learning_rate=0.0014)
        #begin = timer()
        syn = tf.train.SyncReplicasOptimizer(grad, replicas_to_aggregate=num_workers, total_num_replicas=num_workers, use_locking=True)
        #begin_again = timer()
        #elapse = begin_again - begin
        #print("elapse", elapse)
        train_op = syn.minimize(loss, global_step=global_step)
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(out,1),tf.argmax(Y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


  tf.local_variables_initializer()
  tf.global_variables_initializer()

  sync_replicas_hook = syn.make_session_run_hook(is_chief,num_tokens=0)
  stop_hook= tf.train.StopAtStepHook(last_step=1000000)
  hooks = [sync_replicas_hook,stop_hook]

    #tensorboar
    #writer = tf.summary.FileWriter(tboard_summaries)
    #saver = tf.train.Saver()

  print(" Start Training .... ")



  with tf.train.MonitoredTrainingSession( master=server.target, is_chief=is_chief, hooks=hooks) as mon_sess:
    step = 0
    i = 0
    #arr = []
    End = 0
    total_images = 0
    total_time = 0
    stepsum = 0
    while not mon_sess.should_stop():
      Begin = timer()
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      #mnist_ds.batch(batch_size)
      total_images += batch_size
      _, global_step_value = mon_sess.run([train_op, global_step],feed_dict={X: batch_xs, Y: batch_ys})
      i= i + 1
      
      step = step + 1
      start2 = timer()
      oncetime = (start2 - Begin)
      End = End + (start2 - Begin)
      total_time = total_time + oncetime

      #if (step % 100 == 0):
      stepsum = stepsum + 1

      with open('itrTime','a') as out:
        out.write('\n%.10f' % oncetime)

      img = (total_images/total_time)
      cost, acc = mon_sess.run([loss,accuracy], feed_dict={X: mnist.validation.images, Y: mnist.validation.labels})
      print("(stepsum = ", stepsum,"/60000", " cost= ", cost, "accuracy= ", acc, " Time = ", oncetime, " global_step = ", global_step_value)
      with open('Throughput','a') as out:
        out.write('\n%d' % img)      

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