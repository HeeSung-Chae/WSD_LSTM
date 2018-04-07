import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 문장을 읽고, 문장 어절들을 벡터로...

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = tf.Variable(tf.random_normal([n_hidden, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))


x = tf.transpose(x, [1, 0, 2])
x = tf.reshape(x, [-1, n_input])
x = tf.split(0, n_steps, x )

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell( n_hidden, forget_bias=1.0)
outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)
pred = tf.matmul(outputs[-1], weights ) + biases

cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(pred, y))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))

        sess.run(train, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("step : %d, acc: %f" % ( step, acc ))
        step += 1
    print("train complete!")

    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("test accuracy: ", sess.run( accuracy, feed_dict={x: test_data, y: test_label}))



# 듀토리얼 2
# import tensorflow as tf
# import numpy as np
#
# x_data = np.float32(np.random.rand(2, 100))
# y_data = np.dot([1.100, 3.500], x_data) + (-0.500)
#
# print(x_data)
# print()
# print(y_data)
#
# b = tf.Variable(tf.zeros([1]))
#
# W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# y = tf.matmul(W, x_data) + b
#
# print()
# print(b)
# print("===")
# print(W)
# print("===")
# print(y)
#
# loss = tf.reduce_mean(tf.square(y - y_data))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# print()
#
#
# init = tf.initialize_all_variables()
#
# sess = tf.Session()
# sess.run(init)
#
# for step in range(0, 201):
#     sess.run(train)
#     if step % 20 == 0:
#         print(step, sess.run(W), sess.run(b))



# 듀토리얼 1
# state = tf.Variable(0, name="counter")
#
# one = tf.constant(2)
# new_value = tf.add(state, one)
# update = tf.assign(state, new_value)
#
# init_op = tf.initialize_all_variables()
#
# print("state : ", state)
# print("one : ", one)
# print("new_value : ", new_value)
# print("init_op : ", init_op)
# print()
# with tf.Session() as sess:
#     sess.run(init_op)
#     print("run 1 : ", sess.run(state))
#     print()
#     for _ in range(3):
#         sess.run(update)
#         print(sess.run(state))
