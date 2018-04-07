import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

# example
#  text: 'hihello'
# unique chars (vocabulary, voc): h, i, e, l, o
# voc index: h:0, i:1, e:2, l:3, o:4
# h [1 ,0, 0, 0, 0]
# i [0 ,1, 0, 0, 0]
# e [0 ,0, 1, 0, 0]
# l [0 ,0, 0, 1, 0]
# o [0 ,0, 0, 0, 1]

# input  :  h -> i-> h -> e -> l -> l
# output : i -> h -> e -> l -> l -> o

# Data creation
idx2char = ['h', 'i', 'e', 'l', 'o']        # h=0, i=1, e=2, l=3, o=4
x_data = [[0, 1, 0, 2, 3, 3]]       # hihell
x_one_hot = [[[0, 0, 0, 0, 0],      # h 0
                      [0, 1, 0, 0, 0],      # i 1
                      [0, 0, 0, 0, 0],      # h 0
                      [0, 0, 1, 0, 0],      # e 2
                      [0, 0, 0, 1, 0],      # l 3
                      [0, 0, 0, 1, 0]]]     # l 3
y_data = [[1, 0, 2, 3, 3, 4]]       # ihello



# create Cell
hidden_size = 5
input_dim = 5
batch_size = 1
sequence_length = 6

X = tf.placeholder(tf.float32,
                   [batch_size, sequence_length, input_dim])
Y = tf.placeholder(tf.int32, [batch_size, sequence_length])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
# 여기서는 initial_state를 0으로
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)


# Cost: sequence_loss
# logits: 예측값, targets: 출력값(정답)

# # [batch_size, sequence_length]
# y_data = tf.constant([[1,1,1]])
# # [batch_size, sequence_length, emb_dim]
# prediction = tf.constant([[[0.2, 0.7], [0.6, 0.2], [0.2, 0.9]]], dtype=tf.float32)
# # [batch_size * sequence_length]
# weights = tf.constant([[1,1,1]], dtype=tf.float32)

weights = tf.ones([batch_size, sequence_length])

# rnn에서 나온 값을 logits에 바로 넣으면 안 좋음
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs,
                                                 targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)


prediction = tf.argmax(outputs, axis=2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y:y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})

        if(i==199):
            print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

            # print char using dic
            result_str = [idx2char[c] for c in np.squeeze(result)]
            print("\tPrediction str: ", ''.join(result_str))
            print()
            for a, b in zip(result, y_data):
                for c, d in zip(a, b):
                    print(c, d, end="   ")
                    if(c == d):
                        print("true")
                    else:
                        print("false")
            # for a in result:
            #     for b in a:
            #         print(b, type(b))
            #     print()
            #     print(a)
            #     print(type(a))
            # print()
            # for a in y_data:
            #     for b in a:
            #         print(b, type(b))
            #     print(a)
            #     print(type(a))


# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# import numpy as np
#
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#
# learning_rate = 0.001
# training_iters = 100000
# batch_size = 128
# display_step = 10
#
# n_input = 28
# n_steps = 28
# n_hidden = 128
# n_classes = 10
#
# x = tf.placeholder(tf.float32, [None, n_steps, n_input])
# y = tf.placeholder(tf.float32, [None, n_classes])
#
# weights = tf.Variable(tf.random_normal([n_hidden, n_classes]))
# biases = tf.Variable(tf.random_normal([n_classes]))
#
#
# x = tf.transpose(x, [1, 0, 2])
# x = tf.reshpae(x, [-1, n_input])
# x = tf.split(0, n_steps, x )
#
# lstm_cell = tf.nn.rnn_cell.BasicLSTMCell( n_hidden, forget_bias=1.0)
# outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)
# pred = tf.matmul(outputs[-1], weights ) + biases
#
# cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(pred, y))
# train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
# init = tf.initialize_all_variables()
#
# with tf.Session() as sess:
#     sess.run(init)
#     step = 1
#
#     while step * batch_size < training_iters:
#         batch_x, batch_y = mnist.train.next_batch(batch_size)
#         batch_x = batch_x.reshape((batch_size, n_steps, n_input))
#
#         sess.run(train, feed_dict={x: batch_x, y: batch_y})
#         if step % display_step == 0:
#             acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
#             loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
#             print "step : %d, acc: %f" % ( step, acc )
#         step += 1
#     print "train complete!"
#
#     test_len = 128
#     test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
#     test_label = mnist.test.labels[:test_len]
#     print "test accuracy: ", sess.run( accuracy, feed_dict={x: test_data, y: test_label})