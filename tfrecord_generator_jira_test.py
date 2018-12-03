#!/usr/bin/python
# -*- coding: UTF-8 -*-

import glob
import os
import tensorflow as tf
import io
from util import Vocab
import pickle
import numpy as np
from tfrecord_generator_jira import  *


BATCH_SIZE = 2
FILEPATH='./dataset/jira'
TRAIN_TFRECORD_FILE = './dataset/jira/train.tfrecords'
DEV_TFRECORD_FILE = './dataset/jira/dev.tfrecords'
VAL_TFRECORD_FILE = './dataset/jira/val.tfrecords'
MAX_LENGTH = 200


def vocab_test():
    assignee_vocab = load_obj(FILEPATH, 'assignee_vocab')
    mpss_pl_vocab = load_obj(FILEPATH, 'mpss_pl_vocab')
    general_vocab = load_obj(FILEPATH, 'general_vocab')

    #for i in assignee_vocab.index_to_word:
        #print "The %dth index for assignee_vocab is %s"%(i, assignee_vocab.index_to_word[i])

    #for word in assignee_vocab.word_to_index:
        #print "Word %s's index in assignee_vocab is %d"%(word, assignee_vocab.word_to_index[word])

    #for i in mpss_pl_vocab.index_to_word:
        #print "The %dth index for mpss_pl_vocab is %s" % (i, mpss_pl_vocab.index_to_word[i])

    #for word in mpss_pl_vocab.word_to_index:
        #print "Word %s's index in mpss_pl_vocab is %d"%(word, mpss_pl_vocab.word_to_index[word])

    for word in general_vocab.word_to_index:
        print "Word %s's index in general_vocab is %d"%(word, general_vocab.word_to_index[word])

    for i in general_vocab.index_to_word:
        print "The %dth index for general_vocabis %s" % (i, general_vocab.index_to_word[i])



    print "Total length for assignee vocab is %d"%(len(assignee_vocab))
    print "Total length for mpss_pl_vocab is %d"%(len(mpss_pl_vocab))
    print "Total length for general_vocab is %d"%(len(general_vocab))


def _parse_function(example_proto):
  features = {"y": tf.VarLenFeature(tf.int64),
              "x": tf.VarLenFeature(tf.int64)}
  parsed_features = tf.parse_single_example(example_proto, features)
  return tf.sparse_tensor_to_dense(parsed_features["y"]), tf.sparse_tensor_to_dense(parsed_features["x"])


def check_tfrecord_file(tfrecord_file, input_vocab, label_vocab):
    # Check TFRocord file.
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_file)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        if input_vocab:
            x = (example.features.feature['x'].int64_list.value)

        if label_vocab:
            y = (example.features.feature['y'].int64_list.value)
        # data = np.fromstring(data_string, dtype=np.int64)
        print "One item printed out ================================================================"

        if label_vocab:
            print ' '.join(label_vocab.decode_index_list(y))

        if input_vocab:
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

#vocab_test()
check_tfrecord_file(TRAIN_TFRECORD_FILE, label_vocab= mpss_pl_vocab, input_vocab=None)
#check_tfrecord_file(TFRECORD_FILE)