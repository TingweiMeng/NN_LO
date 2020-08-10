import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x_data = 1000
n_data = x_data * x_data
dim = 5

# 0 for 1 dim, 1 for hd, 2 for 1-norm, 3 for inf-norm
eg_no = 1

def min_fn(x):
    return tf.math.reduce_min(x, axis=-1)

# J = -x^2/2.
def J_fn(x):
    val = -tf.reduce_sum(tf.multiply(x,x), axis = -1)/2
    return val

tf.reset_default_graph()

if eg_no == 0:
    v_true = np.array([[-2,0], [0,2], [2,0]])
    b_true = np.array([0.5, -5, 1])
elif eg_no == 1:
    v_true = np.concatenate([np.array([[-2,0,0], [2,-2,-1], [0,2,0]]), np.zeros((3, dim-3))], axis=-1)
    b_true = np.array([0.5, -5, 1])
elif eg_no == 2:
    # H = 1-norm
    for i in range(2**dim):
        bits = "{:0>32b}".format(i)
        bits_arr = np.fromstring(bits[-dim:], dtype=np.uint8) - 48
        vi = bits_arr.astype(np.int) * 2 - 1
        vi = np.expand_dims(vi, axis = 0)
        if i==0:
            v_true = vi
        else:
            v_true = np.concatenate([v_true, vi], axis = 0)
#        print(v_true.shape)
#        print(v_true)
    b_true = np.zeros((2**dim,))
elif eg_no == 3:
    # H = inf-norm
    for i in range(dim):
        zero_tmp = np.zeros((1, dim))
        zero_tmp[0,i] = 1
        vi = np.concatenate([zero_tmp, -zero_tmp], axis = 0)
        if i==0:
            v_true = vi
        else:
            v_true = np.concatenate([v_true, vi], axis = 0)
#        print(v_true.shape)
#        print(v_true)
    b_true = np.zeros((2*dim,))

v_param = tf.Variable(v_true, name = "v0", dtype = tf.float64)
b_param = tf.Variable(b_true, name = "b0", dtype = tf.float64)
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

t_grid0 = np.arange(n_data) * 0 + 0
t_grid = np.expand_dims(t_grid0, axis = -1)

def construct_nn(x_in, t_in):
    t = tf.expand_dims(t_in, -1)
    val0 = tf.subtract(x_in, tf.multiply(t, v_param))
    val1 = J_fn(val0)
    val2 = tf.add(val1, tf.multiply(t_in, b_param))
    y_ = min_fn(val2)
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
# plt.plot(x_grid0, y_val)
#plt.title('formula 2')
plt.savefig('formula2_t0')

sess.close()
