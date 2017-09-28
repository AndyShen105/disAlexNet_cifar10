#-*-coding:UTF-8-*-
from __future__ import print_function

import tensorflow as tf
import sys
import time
import os
import tensorflow as tf
from cifar10 import *
import cifar10 

#get the optimizer
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['GRPC_VERBOSITY_LEVEL']='DEBUG'

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_float("targted_accuracy", 0.5, "targted accuracy of model")
tf.app.flags.DEFINE_string("optimizer", "SGD", "optimizer we adopted")
tf.app.flags.DEFINE_integer("Batch_size", 128, "Batch size")
tf.app.flags.DEFINE_float("Learning_rate", 0.0001, "Learning rate")
tf.app.flags.DEFINE_integer("Epoch", 10, "Epoch")
tf.app.flags.DEFINE_string("imagenet_path", 10, "ImageNet data path")
FLAGS = tf.app.flags.FLAGS

# cluster specification
parameter_servers = sys.argv[1].split(',')
n_PS = len(parameter_servers)
workers = sys.argv[2].split(',')
n_Workers = len(workers)
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# start a server for a specific task
server = tf.train.Server(
    cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index)
	
# config
batch_size = FLAGS.Batch_size
learning_rate = FLAGS.Learning_rate
targted_accuracy = FLAGS.targted_accuracy
Optimizer = FLAGS.optimizer
Epoch = FLAGS.Epoch
imagenet_path = FLAGS.imagenet_path
parameters_initialize(batch_size, learning_rate)

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
		worker_device="/job:worker/task:%d" % FLAGS.task_index,
		cluster=cluster)):
	#More to come on is_chief...
        is_chief = FLAGS.task_index == 0
	# count the number of global steps
	global_step = tf.get_variable('global_step',[],initializer = tf.constant_initializer(0),trainable = False)
	
	# input images
	x, y_ = cifar10.distorted_inputs()
	
	#creat an CNN for cifar10
  	y_conv = cifar10.inference(x)

	# specify cost function
	loss = cifar10.loss(y_conv, y_)

	
	with tf.name_scope('learning_rate'):
	    lr = tf.train.exponential_decay(learning_rate,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
	
	train_op = cifar10.train(loss, global_step)

	# accuracy
	with tf.name_scope('Accuracy'):
	    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	init_op = tf.global_variables_initializer()
	variables_check_op=tf.report_uninitialized_variables()

    	sess_config = tf.ConfigProto(
        	allow_soft_placement=True,
        	log_device_placement=False,
        	device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])
    sv = tf.train.Supervisor(is_chief=is_chief,
			     init_op=init_op,
                             global_step=global_step)
    server_grpc_url = "grpc://" + workers[FLAGS.task_index]
    state = False
    with sv.prepare_or_wait_for_session(server_grpc_url, config=sess_config) as sess:
	while(not state):
            uninitalized_variables=sess.run(variables_check_op)
	    if(len(uninitalized_variables.shape) == 1):
		state = True
	
	step = 0
	cost = 0
	final_accuracy = 0
	start_time = time.time()
	batch_time = time.time()
	epoch_time = time.time()
	while (not sv.should_stop()):
	    #Read batch_size data
	    for e in range(Epoch):
		for i in range(int(num_batches_per_epoch)):
                    _, cost, step = sess.run([train_op, loss, global_step])
	    	    print("Step: %d," % (step+1), 
			#        " Accuracy: %.4f," % final_accuracy,
			        " Loss: %f" % cost,
			        " Bctch_Time: %fs" % float(time.time()-batch_time))
	    	    batch_time = time.time()
		

	    	print("Epoch: %d," % (e+1), 
			" Accuracy: %.4f," % final_accuracy,
			" Loss: %f" % cost,
			" Epoch_Time: %fs" % float(time.time()-epoch_time),
			" Tolal_Time: %fs" % float(time.time()-start_time))
		
		
	#index, sum_step, total_time, cost, final_accuracy    
	final_accuracy = sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
	re = str(n_PS) + '-' + str(n_Workers) + '-' + str(FLAGS.task_index) + ',' + str(step) + ',' + str(float(time.time()-start_time)) + ',' + str(cost) + ',' + str(final_accuracy)
        writer = open("re_2_"+Optimizer+".csv", "a+")
	writer.write(re+"\r\n")
	writer.close()
    sv.stop 
