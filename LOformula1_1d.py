import random
import numpy as np
import tensorflow as tf
import time
import scipy.io
import matplotlib.pyplot as plt

start = time.time()

width = 3 # number of neurons
n_data = 1000
dim = 1

def min_fn(x):
    return tf.math.reduce_min(x, axis=-1)

# L = -x-1/2 if x<-1; x^2/2 if -1<=x<=2; 2x-2 if x>2.
def L_fn(x):
    val1 = -x - 0.5
    val2 = tf.multiply(x,x)/2
    val3 = 2*x - 2
    flag1 = 1 - tf.sign(tf.maximum(x+1, 0))
    flag3 = tf.sign(tf.maximum(x-2, 0))
    flag2 = 1 - flag1 - flag3
    val = tf.multiply(flag1, val1)
    val = tf.add(val, tf.multiply(flag2, val2))
    val = tf.add(val, tf.multiply(flag3, val3))
    return tf.squeeze(val, -1)

tf.reset_default_graph()

u_true = np.array([[-2], [0], [2]])
a_true = np.array([-0.5, 0, -1])
u_param = tf.Variable(u_true, name = "u0", dtype = tf.float64)
a_param = tf.Variable(a_true, name = "a0", dtype = tf.float64)


x_placeholder = tf.placeholder(tf.float64, shape=(n_data, 1, dim))
t_placeholder = tf.placeholder(tf.float64, shape=(n_data, 1))

def construct_nn(x_in, t_in):
    val0 = tf.subtract(x_in, u_param)
    val1 = tf.div(val0, tf.expand_dims(t_in, -1))
    val2 = L_fn(val1)
    val3 = tf.add(tf.multiply(val2, t_in), a_param)
    y_ = min_fn(val3)
    return y_

y_nn = construct_nn(x_placeholder, t_placeholder)

sess = tf.Session()

# initialization
sess.run(tf.global_variables_initializer())

x_grid0 = np.arange(n_data) * (10.0 / n_data) - 5
t_grid0 = np.arange(n_data) * 0 + 3
x_grid = np.expand_dims(x_grid0, axis = -1)
x_grid = np.expand_dims(x_grid, axis = -1)
t_grid = np.expand_dims(t_grid0, axis = -1)

y_val = sess.run(y_nn, {x_placeholder: x_grid, t_placeholder: t_grid})

plt.plot(x_grid0, y_val)
plt.title('formula1 t3')
plt.savefig('formula1_t3')

sess.close()
