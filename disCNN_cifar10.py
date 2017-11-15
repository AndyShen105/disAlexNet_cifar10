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
tf.app.flags.DEFINE_integer("Batch_size", 100, "Batch size")
tf.app.flags.DEFINE_float("Learning_rate", 0.0001, "Learning rate")
tf.app.flags.DEFINE_integer("Epoch", 200, "Epoch")
tf.app.flags.DEFINE_string("imagenet_path", 100, "ImageNet data path")
tf.app.flags.DEFINE_integer("n_intra_threads", 16, "n_intra_threads")
tf.app.flags.DEFINE_integer("n_inter_threads", 16, "n_inter_threads")
FLAGS = tf.app.flags.FLAGS


# config
batch_size = FLAGS.Batch_size
learning_rate = FLAGS.Learning_rate
targted_loss = FLAGS.targted_loss
Optimizer = FLAGS.optimizer
Epoch = FLAGS.Epoch
n_intra_threads = FLAGS.n_intra_threads
n_inter_threads = FLAGS.n_inter_threads
imagenet_path = FLAGS.imagenet_path
parameters_initialize(batch_size, learning_rate)
n_batches_per_epoch = int(50000/batch_size)


# cluster specification
parameter_servers = sys.argv[1].split(',')
n_PS = len(parameter_servers)
workers = sys.argv[2].split(',')
n_Workers = len(workers)
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})
server_config = tf.ConfigProto(
                intra_op_parallelism_threads=n_intra_threads,
                inter_op_parallelism_threads=n_inter_threads)

# start a server for a specific task
server = tf.train.Server(
    cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index,
    config=server_config)
	
if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    # Between-graph replicationee
    with tf.device(tf.train.replica_device_setter(
		worker_device="/job:worker/task:%d" % FLAGS.task_index,
		cluster=cluster)):
	#More to come on is_chief...
        is_chief = FLAGS.task_index == 0
	# count the number of global steps
	global_step = tf.get_variable('global_step',[],initializer = tf.constant_initializer(0),trainable = False)
	start_time = tf.Variable(time.time() ,dtype = tf.float64, trainable = False)
	start_copy = tf.placeholder(tf.float64)
	update = tf.assign(start_time, start_copy, validate_shape=None, use_locking=False)
	
	# input images
	x, y_ = cifar10.distorted_inputs(batch_size)
	
	#creat an CNN for cifar10
  	y_conv = cifar10.inference(x, batch_size)
 
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
	begin_time = time.time()
	batch_time = time.time()
	n = 0
	cost = 1000000.0
	step = 1
	count = 0
	result_data = open("/root/ex_result/baseline/cnn_result_2.csv", "a+")
	while (not sv.should_stop()) and (step <= n_batches_per_epoch * Epoch and cost>=targted_loss):

            _, cost, step = sess.run([train_op, loss, global_step])
	    '''
	    flag1 = step % n_batches_per_epoch
	    flag2 = (step - 1) % n_batches_per_epoch
	    flag3 = (step - 2) % n_batches_per_epoch
	    if (flag1!=0 or flag2==0 or flag3==0):
		epoch_time = sess.run(start_time)
		print (epoch_time)
		update_time = float(time.time())
		Epoch_Time = float(update_time - epoch_time)
		sess.run(update, feed_dict={start_copy: update_time})
		ith_epoch = int(step / n_batches_per_epoch)
	    
	    print("Epoch: %d," % ith_epoch, 
			" Loss: %f" % cost,
			" Epoch_Time: %fs" % Epoch_Time,
			" Tolal_Time: %fs" % float(time.time()-begin_time))
	    '''
	    process_data = open("/root/ex_result/baseline/cnn_"+str(learning_rate)+"_"+str(batch_size)+"_process_2.csv", "a+")
	    line = str(step+1)+","+str(n_Workers) + ','+str(n_intra_threads)+","+ str(cost) + ',' + str(time.time())
            process_data.write(line+"\r\n")
	    process_data.close()
	    print("Step: %d," % (step+1),
                            " Loss: %f" % cost,
                            " Bctch_Time: %fs" % float(time.time()-batch_time))
            batch_time = time.time()	
    	total_time = time.time()-begin_time
	re = str(step+1)+","+str(n_Workers) + ','+str(n_intra_threads)+","+ str(cost) + ','+str(learning_rate)+","+str(batch_size)+","+ str(total_time)
	result_data.write(re+"\r\n")
	result_data.close()
    sv.stop 
