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
tf.app.flags.DEFINE_float("targted_loss", 0.05, "targted accuracy of model")
tf.app.flags.DEFINE_string("optimizer", "SGD", "optimizer we adopted")
tf.app.flags.DEFINE_integer("Batch_size", 128, "Batch size")
tf.app.flags.DEFINE_float("Learning_rate", 0.0001, "Learning rate")
tf.app.flags.DEFINE_integer("Epoch", 1, "Epoch")
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
targted_loss = FLAGS.targted_loss
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
 	'''
	# specify cost function
	loss = cifar10.loss(y_conv, y_)
	
	train_op = cifar10.train(loss, global_step)
	'''
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	# specify optimizer
	with tf.name_scope('train'):
	    grad_op = get_optimizer( "Adam", learning_rate)
	    train_op = grad_op.minimize(loss, global_step=global_step)
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
	n = 0
	cost = 1.0
	step = 0
	result_data = open("/root/result/result_pic.csv", "a+")
	process_data = open("/root/result/process_data.csv", "a+")
	while (not sv.should_stop()) and (step <= num_batches_per_epoch * Epoch):
	    if (cost <= targted_loss):
		break
            _, cost, step = sess.run([train_op, loss, global_step])
	    print("Step: %d," % (step+1), 
			    " Loss: %f" % cost,
			    " Bctch_Time: %fs" % float(time.time()-batch_time))
	    batch_time = time.time()
	    if ((step+1) % int(num_batches_per_epoch) == 0):
		Epoch_Time = float(time.time()-epoch_time)
		n=n+1
	    	print("Epoch: %d," % (n), 
			" Loss: %f" % cost,
			" Epoch_Time: %fs" % Epoch_Time,
			" Tolal_Time: %fs" % float(time.time()-start_time))
		epoch_time = time.time()
		line = str(n)+","+str(n_PS) + ',' + str(n_Workers) +",RoundRobinStrategy,asynchronous," + str(Optimizer) + ',' + str(batch_size) + ','+"learning_rate"+ ',' + str(cost) + ',' + str(Epoch_Time)
        	process_data.write(line+"\r\n")

    	total_time = time.time()-start_time
        result = str(n_PS) + ',' + str(n_Workers) +",RoundRobinStrategy,asynchronous," + str(Optimizer) + ',' + str(batch_size) + ','+"learning_rate"+ ',' + str(cost) + ',' + str(total_time)
	result_data.write(result+"\r\n")
	result_data.close()
	process_data.close()
    sv.stop 
