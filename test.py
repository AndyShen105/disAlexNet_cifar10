import tensorflow as tf
import time

start = tf.Variable(time.time(), trainable = False)
temp=start
start=time.time
s=start-temp
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(s)
