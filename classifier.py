#!/usr/bin/env python

import tensorflow as tf
import numpy as np


feature_colume1 = tf.SparseTensor(indices=[[0, 0], [1, 8], [2, 6]], values=[1, 1, 1], shape=[3, 10])



vocabulary_size = 10
embedding_size = 2
var = np.array([[0.0, 1.0], [4.0, 9.0], [16.0, 25.0], [36.0, 49.0], [64.0, 81.0], [0.0, 1.0], [4.0, 9.0], [16.0, 25.0], [36.0, 49.0], [64.0, 81.0]])
#embeddings = tf.Variable(tf.ones([vocabulary_size, embedding_size]))
embeddings = tf.Variable([[0.0, 1.0], [4.0, 9.0], [16.0, 25.0], [36.0, 49.0], [64.0, 81.0], [0.0, 1.0], [4.0, 9.0], [16.0, 25.0], [36.0, 49.0], [64.0, 81.0]])
bias = tf.Variable([1.0, 2.0])

batch_size = 3
feature_number = 1

train_inputs = tf.placeholder(tf.int32, shape=[batch_size, feature_number])


embed = tf.nn.embedding_lookup(embeddings, train_inputs)
#embed = tf.nn.embedding_lookup(embeddings, np.array([[3, 6, 9], [2, 3]]))

logis = tf.add(embed, bias)


batch_data = np.array([[0], [8], [6]])


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    print(sess.run(embed, feed_dict={train_inputs: batch_data}))
    
    print(sess.run(logis, feed_dict={train_inputs: batch_data}))
    #print(sess.run(embeddings))
