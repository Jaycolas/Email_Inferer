#!/usr/bin/python
# -*- coding: UTF-8 -*-

import glob
import io
import tensorflow as tf
import os
import numpy as np
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
import codecs
import json

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
    self.most_cnt_one_word = 0
    self.add_word(self.unknown, count=5000)
    self.add_word(self.sos, count=5000)
    self.add_word(self.eos, count=5000)
    self.add_word(self.padding, count=5000)


  def add_word(self, word, count=1):
    if word not in self.word_to_index:
      index = len(self.word_to_index)
      self.word_to_index[word] = index
      self.index_to_word[index] = word
      self.vocab_len+=1
    self.word_freq[word] += count

    #Record the most count of one word
    if self.word_freq[word] > self.most_cnt_one_word:
      self.most_cnt_one_word = self.word_freq[word]

  def remove_word(self, word):
    if word in self.word_to_index:
      print 'Deleting word %s in dictionary'%(word)
      index = self.word_to_index[word]
      del self.word_freq[word]
      del self.word_to_index[word]
      del self.index_to_word[index]
      self.vocab_len-=1

  def filter_dictionary(self, lower_threshold):
    filter_cnt = 0
    for word, freq in self.word_freq.items():
      if freq <= lower_threshold:
        self.remove_word(word)
        filter_cnt+=1
    print("Filtered {} out of {} total words".format(filter_cnt, self.__len__()))

  def word_distribution(self):
    word_count_list = np.zeros(self.most_cnt_one_word+1)
    for word, count in self.word_freq.items():
      word_count_list[count]+=1

    x = np.arange(self.most_cnt_one_word+1)
    y = word_count_list
    #colors = np.random.rand(N)
    #area = (30 * np.random.rand(N)) ** 2  # 0 to 15 point radii

    plt.scatter(x[0:300], y[0:300])
    plt.show()

  #After certain kinds of filtering the index might not be sequential, need to reorder it.
  def reorder_dictionary(self):
    i = 0
    #Please note that python will not allow you modify dictionary when you iterate it, so using enumerate won't work
    for index in self.index_to_word.keys():
        if index > i:
            #when index>i you found out one jump value
            word = self.index_to_word[index]   #first you get the word of the unsequential word
            self.word_to_index[word] = i       #You set the word_to_index with the sequential index
            self.index_to_word[i] = word       #You create a new item for index_to_word with index = i
            del self.index_to_word[index]      #Delete the same value with the jump index
        i+=1

  def construct(self, words):
    for word in words:
      self.add_word(word)
    self.total_words = float(sum(self.word_freq.values()))
    #self.vocab_len = len(self.word_freq)
    #print('{} total words with {} uniques'.format(self.total_words, len(self.word_freq)))

  def encode(self, word):
    if word not in self.word_to_index:
      word = self.unknown
    return self.word_to_index[word]

  def encode_word_list(self, word_list):
    return map(lambda x:self.encode(x), word_list)

  def decode(self, index):
    if index not in self.index_to_word:
        index = self.unknown
    return self.index_to_word[index]

  def decode_index_list(self, index_list):
    if index_list is None:
        return 0

    return map(lambda x: self.decode(x), index_list)


  def __len__(self):
    return len(self.word_freq)

def tokenize_helper(txt):
    if txt is not None:
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
        #print index
        index = index[index != 0]
        #print index
        label_vector = np.zeros(int(len(label_vocab)))
        label_vector[index]=1
        ret_dense_vector.append(label_vector)

    return ret_dense_vector

def label_vector_to_index(label_batch, label_vocab):
    #print label_batch
    #print np.shape(label_batch)
    #if np.shape(label_batch)[0]==1:
        #label_batch=np.squeeze(label_batch, axis=0)
    #print np.shape(label_batch)
    #print np.shape(label_batch)
    ret_dense_vector = []
    ret_email_addr = []
    for label in label_batch:
        #print label
        index_list = np.nonzero(label)
        #index_list should be a tuple
        #print index_list
        #print np.shape(index_list)
        print index_list

        email_list = label_vocab.decode_index_list(index_list[0].tolist())  #Change the numpy array to list
        ret_dense_vector.append(index_list[0])
        ret_email_addr.append(email_list)
    return ret_dense_vector, ret_email_addr

def get_device_str(device_id, num_gpus):
    """Return a device string for multi-GPU setup."""
    if num_gpus == 0:
        return "/cpu:0"
    device_str_output = "/gpu:%d" % (device_id % num_gpus)
    return device_str_output


def save_obj(filepath, obj, name):
    with open(filepath + '/vocab/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filepath, name):
    with open(filepath + '/vocab/' + name + '.pkl', 'rb') as f:
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


def create_hparams(flags):
  """Create training hparams."""
  return tf.contrib.training.HParams(
      # Data
      dev_sample_percentage=flags.dev_sample_percentage,

      # Model parameters
      embedding_size=flags.embedding_size,
      filter_sizes=flags.filter_sizes,
      num_filters=flags.num_filters,
      dropout_keep_prob=flags.dropout_keep_prob,
      sequence_length=flags.sequence_length,
      learning_rate=flags.learning_rate,
      l2_reg_lambda=flags.l2_reg_lambda,
      is_multiclass=flags.is_multiclass,

      # Training Parameters
      batch_size=flags.batch_size,
      num_epochs=flags.num_epochs,
      evaluate_every=flags.evaluate_every,
      #checkpoint_every=flags.checkpoint_every,
      num_checkpoints=flags.num_checkpoints,
      restore_checkpoint = flags.restore_checkpoint,
      warmup_scheme=flags.warmup_scheme,
      decay_scheme=flags.decay_scheme,
      num_train_steps=flags.num_train_steps,
      warmup_step = flags.warmup_step,
      checkpoint_every_epoch=flags.checkpoint_every_epoch,
      num_gpus=flags.num_gpus,

      # Misc Parameters
      log_device_placement=flags.log_device_placement,
      allow_soft_placement=flags.allow_soft_placement,
      debug=flags.debug
  )

def create_or_load_hparams(hparams_file, default_hparams):
  """Create hparams or load hparams from out_dir."""
  hparams = load_hparams(hparams_file)
  if not hparams:
    hparams = default_hparams

  return hparams


def load_hparams(hparams_file):
  """Load hparams from an existing model directory."""
  if tf.gfile.Exists(hparams_file):
    print("# Loading hparams from %s" % hparams_file)
    with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
      try:
        hparams_values = json.load(f)
        hparams = tf.contrib.training.HParams(**hparams_values)
      except ValueError:
        print("  can't load hparams file")
        return None
    return hparams
  else:
    print("Nothing has been loaded from %s" % hparams_file)
    return None

def print_hparams(hparams, skip_patterns=None, header=None):
  """Print hparams, can skip keys based on pattern."""
  if header: print("%s" % header)
  values = hparams.values()
  for key in sorted(values.keys()):
    if not skip_patterns or all(
        [skip_pattern not in key for skip_pattern in skip_patterns]):
      print("  %s=%s" % (key, str(values[key])))