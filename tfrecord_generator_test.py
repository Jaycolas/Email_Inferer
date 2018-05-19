#!/usr/bin/python
# -*- coding: UTF-8 -*-

import glob
import os
import tensorflow as tf
import io
from util import Vocab
import pickle
import numpy as np
from tfrecord_generator import  *
from sklearn.preprocessing import MultiLabelBinarizer


BATCH_SIZE = 2
TFRECORD_FILE = './dataset/email/train.tfrecords'
MAX_LENGTH=200


def vocab_test():
    input_vocab = load_obj('input_vocab')
    label_vocab = load_obj('label_vocab')

    for i in input_vocab.index_to_word:
        print "The %dth index for input vocab is %s"%(i, input_vocab.index_to_word[i])
    '''
    for i in label_vocab.index_to_word:
        print "The %dth index for this vocab is %s" % (i, label_vocab.index_to_word[i])

    for word in label_vocab.word_to_index:
        print "Word %s's index is %d"%(word, label_vocab.word_to_index[word])'''

    print "Total length for label vocab is %d"%(label_vocab.vocab_len)
    print "Total length for input vocab is %d"%(input_vocab.vocab_len)
def _parse_function(example_proto):
  features = {"y": tf.VarLenFeature(tf.int64),
              "x": tf.VarLenFeature(tf.int64)}
  parsed_features = tf.parse_single_example(example_proto, features)
  return tf.sparse_tensor_to_dense(parsed_features["y"]), tf.sparse_tensor_to_dense(parsed_features["x"])


def check_tfrecord_file(tfrecord_file):
    # Check TFRocord file.
    input_vocab = load_obj('input_vocab')
    label_vocab = load_obj('label_vocab')
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_file)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        x = (example.features.feature['x'].int64_list.value)
        y = (example.features.feature['y'].int64_list.value)
        # data = np.fromstring(data_string, dtype=np.int64)
        print "One item printed out ================================================================"
        print ' '.join(label_vocab.decode_index_list(y))
        print ' '.join(input_vocab.decode_index_list(x))

def padding_max_cut(input_tensor, padded_value, max_length):
    #Cut all dimension of input tensor to be under the same max_length
    #Pad the end if it can't reach the max_length

    #input_tensor : batch_size, variable sequence
    size = tf.shape(input_tensor)
    len = size[0]

    return tf.cond(len > max_length,
            lambda : input_tensor[0:max_length],
            lambda : tf.concat(input_tensor,tf.constant(padded_value, shape=[max_length-len])))


def tfRecordTest():
    dataset = tf.data.TFRecordDataset(TFRECORD_FILE)
    dataset = dataset.map(_parse_function)
    #dataset = dataset.repeat()
    dataset = dataset.filter(lambda y,x : tf.less(tf.shape(x)[0], MAX_LENGTH) )
    #dataset = dataset.batch(1)
    dataset = dataset.padded_batch(4, padded_shapes=([None],[MAX_LENGTH]))
    iterator = dataset.make_one_shot_iterator()
    y,x = iterator.get_next()
    #zeros = tf.zeros_like(y)
    #ones = tf.ones_like(y)
    #y = tf.not_equal(y, 0)
    #y = tf.boolean_mask(y,mask=y_mask, name='label_boolean_mask')
    #y = tf.SparseTensor(indices=y, values=tf.constant(1,dtype=tf.int64), dense_shape=tf.shape(y))



    label_vocab = load_obj('label_vocab')
    #x = padding_max_cut(x,input_vocab.word_to_index[input_vocab.padding], MAX_LENGTH)


    length_count = []

    with tf.Session() as sess:
        #sess.run(iterator.initializer)
        while True:
            try:
                #print tf.shape(x).eval()
                print tf.shape(y).eval()
                out_y,_=sess.run([y,x])
                print out_y
                out_y = index_to_label_vector(out_y,label_vocab)
                print out_y

                #mb = MultiLabelBinarizer()
                #out_y = mb.fit_transform(out_y)
                print "New items decoded ============================"
                #print out_y
                #length_count.append(out_x.size)
            except tf.errors.OutOfRangeError:
                break

        #print 'The max is %d'%(np.max(length_count))
        #print 'The min is %d'%(np.min(length_count))
        #print 'The mean is %d'%(np.mean(length_count))


def index_to_label_vector(index_batch, label_vocab):
    ret_dense_vector = []

    for index in index_batch:
        index = index[index!=0]
        label_vector = np.zeros(int(label_vocab.total_words))
        label_vector[index]=1
        ret_dense_vector.append(label_vector)

    return ret_dense_vector

def test_dense_vector(dense_vector, index, total_length):
    print "The sparse vector is: ", index
    for i in index:
        print "The %dth value is %d"%(i, dense_vector[i])





vocab_test()
#tfRecordTest()
#check_tfrecord_file(TFRECORD_FILE)