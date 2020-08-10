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

# J = -x^2/2.
def J_fn(x):
    val = -tf.multiply(x,x)/2
    return tf.squeeze(val, -1)

tf.reset_default_graph()

v_true = np.array([[-2], [0], [2]])
b_true = np.array([0.5, -5, 1])
v_param = tf.Variable(v_true, name = "v0", dtype = tf.float64)
b_param = tf.Variable(b_true, name = "b0", dtype = tf.float64)


x_placeholder = tf.placeholder(tf.float64, shape=(n_data, 1, dim))
t_placeholder = tf.placeholder(tf.float64, shape=(n_data, 1))

def construct_nn(x_in, t_in):
    t = tf.expand_dims(t_in, -1)
    val0 = tf.subtract(x_in, tf.multiply(t, v_param))
    val1 = J_fn(val0)
    val2 = tf.add(val1, tf.multiply(t_in, b_param))
    y_ = min_fn(val2)
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
plt.title('formula2 t3')
plt.savefig('formula2_t3')

sess.close()
