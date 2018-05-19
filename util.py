#!/usr/bin/python
# -*- coding: UTF-8 -*-

import glob
import io
import tensorflow as tf
import os
import numpy as np
from collections import defaultdict
import pickle

__author__ = 'Jaycolas'


class Vocab(object):
  def __init__(self):
    self.word_to_index = {}
    self.index_to_word = {}
    self.word_freq = defaultdict(int)
    self.total_words = 0
    self.unknown = '<UNK>'
    self.sos = '<SOS>'
    self.eos = '<EOS>'
    self.padding = '<PAD>'
    self.vocab_len = 0
    self.add_word(self.unknown, count=1)
    self.add_word(self.sos, count=1)
    self.add_word(self.eos, count=1)
    self.add_word(self.padding, count=1)

  def add_word(self, word, count=1):
    if word not in self.word_to_index:
      index = len(self.word_to_index)
      self.word_to_index[word] = index
      self.index_to_word[index] = word
    self.word_freq[word] += count

  def construct(self, words):
    for word in words:
      self.add_word(word)
    self.total_words = float(sum(self.word_freq.values()))
    self.vocab_len = len(self.word_freq)
    print '{} total words with {} uniques'.format(self.total_words, len(self.word_freq))

  def encode(self, word):
    if word not in self.word_to_index:
      word = self.unknown
    return self.word_to_index[word]

  def encode_word_list(self, word_list):
    return map(lambda x:self.encode(x),word_list)

  def decode(self, index):
    if index not in self.index_to_word:
        index = self.unknown
    return self.index_to_word[index]

  def decode_index_list(self, index_list):
    if index_list is None:
        return 0

    if len(index_list)==1:
        return self.decode(index_list)
    else:
        return map(lambda x:self.decode(x), index_list)


  def __len__(self):
    return len(self.word_freq)

def tokenize_helper(txt):
    txt = txt.replace(':', ' ')
    #txt = txt.replace('-', ' ')
    txt = txt.replace('/', ' ')
    txt = txt.replace('>', ' ')
    txt = txt.replace('<', ' ')
    txt = txt.replace('?', ' ')
    txt = txt.replace('\\', ' ')
    txt = txt.replace('*', ' ')
    txt = txt.replace('|', ' ')
    txt = txt.replace('#', ' ')
    txt = txt.replace('$', ' ')
    txt = txt.replace('!', ' ')
    txt = txt.replace('\'', ' ')
    txt = txt.replace('(', ' ')
    txt = txt.replace(')', ' ')
    txt = txt.replace(',', ' ')
    txt = txt.replace(';', ' ')
    txt = txt.replace('\"', ' ')
    txt = txt.replace('\'', ' ')
    txt = txt.replace('&', ' ')
    txt = txt.replace('[',' ')
    txt = txt.replace(']',' ')
    txt = txt.replace('[',' ')
    txt = txt.replace('@',' ')
    txt = txt.replace('{',' ')
    txt = txt.replace('}',' ')
    txt = txt.replace('~',' ')
    txt = txt.replace('qti.qualcomm.com',' ')
    return txt


def index_to_label_vector(index_batch, label_vocab):
    ret_dense_vector = []

    for index in index_batch:
        index = index[index!=0]
        label_vector = np.zeros(int(label_vocab.vocab_len))
        label_vector[index]=1
        ret_dense_vector.append(label_vector)

    return ret_dense_vector

def label_vector_to_index(label_batch, label_vocab):
    #print label_batch
    print np.shape(label_batch)
    if np.shape(label_batch)[0]==1:
        label_batch=np.squeeze(label_batch, axis=0)
    #print np.shape(label_batch)
    print np.shape(label_batch)
    ret_dense_vector = []
    ret_email_addr = []
    for label in label_batch:
        #print label
        index_list = np.nonzero(label)
        #print index_list
        print np.shape(index_list)
        if 1 in np.shape(index_list):
            index_list = np.squeeze(index_list)
        email_list = label_vocab.decode_index_list(index_list)
        ret_dense_vector.append(index_list)
        ret_email_addr.append(email_list)
    return ret_dense_vector, ret_email_addr


def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def for_grad_var_names(var_name):
    transfer_name = var_name.replace('/','_')
    transfer_name = var_name.replace(':','_')
    return transfer_name

def get_data_from_files(dataset, input_fname_pattern, dictionary):
    docFiles = glob.glob(os.path.join("./dataset", dataset, input_fname_pattern))
    for file in docFiles:
        # global dictionary
        # print "loading file, ", file
        fd = io.open(file, mode='r', encoding="ISO-8859-1")
        txt = fd.read()
        # self.dictArray.append(self.tokenizer.tokenize(txt.lower()))
        txt = tokenize_helper(txt)

        for word in txt:
            yield word

def ptb_iterator(raw_data,batch_size,num_steps):
  # Pulled from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L82
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)


def variable_summaries(var, var_name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.variable_scope(var_name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)