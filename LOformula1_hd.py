import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x_data = 1000
n_data = x_data * x_data
dim = 10

def min_fn(x):
    return tf.math.reduce_min(x, axis=-1)

# L = max(|x|_2 - 1, 0)
def L_fn(x):
    val = tf.maximum(tf.norm(x, ord=2, axis=-1)-1, 0)
    return val

tf.reset_default_graph()

u_true = np.concatenate([np.array([[-2,0,0], [2,-2,-1], [0,2,0]]), np.zeros((3, dim-3))], axis=-1)
#u_true = np.array([[-2], [0], [2]])
a_true = np.array([-0.5, 0, -1])
u_param = tf.Variable(u_true, name = "u0", dtype = tf.float64)
a_param = tf.Variable(a_true, name = "a0", dtype = tf.float64)
x_placeholder = tf.placeholder(tf.float64, shape=(n_data, 1, dim))
t_placeholder = tf.placeholder(tf.float64, shape=(n_data, 1))

x_grid0 = np.arange(x_data) * (10.0 / x_data) - 5
y_grid0 = np.arange(x_data) * (10.0 / x_data) - 5
X, Y = np.meshgrid(x_grid0, y_grid0)
X_hd = np.expand_dims(X.flatten(), axis = -1)
X_hd = np.expand_dims(X_hd, axis = -1)
Y_hd = np.expand_dims(Y.flatten(), axis = -1)
Y_hd = np.expand_dims(Y_hd, axis = -1)
x_grid = np.concatenate([X_hd,Y_hd,np.zeros((n_data, 1, dim-2))], axis = -1)

t_grid0 = np.arange(n_data) * 0 + 0.000001
t_grid = np.expand_dims(t_grid0, axis = -1)

def construct_nn(x_in, t_in):
    val0 = tf.subtract(x_in, u_param)
    val1 = tf.div(val0, tf.expand_dims(t_in, -1))
    val2 = L_fn(val1)
    val3 = tf.add(tf.multiply(val2, t_in), a_param)
    y_ = min_fn(val3)
    return y_

y_nn = construct_nn(x_placeholder, t_placeholder)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
y_val = sess.run(y_nn, {x_placeholder: x_grid, t_placeholder: t_grid})
y_mat = y_val.reshape((x_data, x_data))

a = plt.contourf(X, Y, y_mat, 10)
plt.colorbar(a)
plt.xlabel('x1')
plt.ylabel('x2')
#plt.plot(x_grid0, y_val)
#plt.title('f1')
plt.savefig('formula1_t1em6')

sess.close()