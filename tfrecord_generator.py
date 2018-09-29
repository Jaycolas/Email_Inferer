#!/usr/bin/python
# -*- coding: UTF-8 -*-

import glob
import os
import tensorflow as tf
import io
from util import Vocab
import pickle
import random

__author__ = 'Jaycolas'

FILEPATH='./dataset/email'
input_fname_pattern='*.txt'
TRAIN_TFRECORD_FILE = os.path.join(FILEPATH, 'train.tfrecords')
DEV_TFRECORD_FILE = os.path.join(FILEPATH, 'dev.tfrecords')
VAL_TFRECORD_FILE = os.path.join(FILEPATH, 'val.tfrecords')
DEV_SAMPLE_PER = 0.2
VAL_SAMPLE_PER = 0.1
doc_file_list = glob.glob(os.path.join(FILEPATH, input_fname_pattern))
LOWER_DIC_FILTER_THRESHOLD = 0

def save_obj(obj, name):
    with open('/obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def buildVocabforInput(file_list):
    print "Building vocabulary for input"
    input_vocab = Vocab()
    for file in file_list:
        fd = io.open(file, mode='r', encoding="ISO-8859-1")
        # When we store the data, first line is for labels
        x_lines = fd.readlines()[1:]
        x_txt = reduce(lambda x,y:x+y, x_lines).split()
        #print x_txt
        input_vocab.construct(x_txt)

    input_vocab.filter_dictionary(lower_threshold=LOWER_DIC_FILTER_THRESHOLD)

    return input_vocab

def buildVocabforLabel(file_list):
    print "Building vocabulary for label"
    label_vocab = Vocab()
    for file in file_list:
        fd = io.open(file, mode='r', encoding="ISO-8859-1")
        #When we store the data, first line is for labels
        y_txt = fd.readlines()[0].split()
        print y_txt
        label_vocab.construct(y_txt)

    label_vocab.filter_dictionary(lower_threshold=LOWER_DIC_FILTER_THRESHOLD)

    return label_vocab


def split_train_dev_val(file_list, dev_per, val_per):
    #Firstly need to check the validity of each input percentage.
    assert dev_per>0 and dev_per<1
    assert val_per>0 and val_per<1
    assert dev_per+val_per<1

    train_per = 1-dev_per-val_per

    #Randomly shuffled the total file list
    shuffled_list = random.sample(file_list, len(file_list))
    print shuffled_list
    total_cnt = len(shuffled_list)
    print "total cnt = %d"%(total_cnt)
    train_len = int(total_cnt * train_per)
    print "training samples' number is %d"%(train_len)
    dev_len = int(total_cnt * dev_per)
    print "dev samples' number is %d" % (dev_len)
    val_len = total_cnt - train_len - dev_len
    print "val samples' number is %d" % (val_len)

    train_file_list = shuffled_list[0:train_len]
    dev_file_list = shuffled_list[train_len:train_len+dev_len]
    val_file_list = shuffled_list[train_len+dev_len: total_cnt]

    return train_file_list, dev_file_list, val_file_list




def writeTfRecordData(file_list, input_vocab, label_vocab, tf_record_file):
    writer = tf.python_io.TFRecordWriter(tf_record_file)
    for file in file_list:
        fd = io.open(file, mode='r', encoding="ISO-8859-1")
        # When we store the data, first line is for labels
        lines = fd.readlines()
        y_txt = lines[0]
        x_lines = lines[1:]
        #print x_lines
        if y_txt and x_lines:
            x_txt = reduce(lambda x,y:x+y, x_lines)
            y = label_vocab.encode_word_list(y_txt.split())
            x = input_vocab.encode_word_list(x_txt.split())
        else:
            print "Either y_txt or x_lines is NULL"
            continue

        example = tf.train.Example(features=tf.train.Features(feature=
            {'y':tf.train.Feature(int64_list=tf.train.Int64List(value=y)),
             'x':tf.train.Feature(int64_list=tf.train.Int64List(value=x))}))

        writer.write(example.SerializeToString())

    writer.close()


def tfrecord_main():
    input_vocab = buildVocabforInput(doc_file_list)
    save_obj(input_vocab,'input_vocab')
    label_vocab = buildVocabforLabel(doc_file_list)
    save_obj(label_vocab, 'label_vocab')
    train_list, dev_list, val_list = split_train_dev_val(doc_file_list,DEV_SAMPLE_PER,VAL_SAMPLE_PER)

    writeTfRecordData(train_list,input_vocab,label_vocab,TRAIN_TFRECORD_FILE)
    writeTfRecordData(dev_list,input_vocab,label_vocab,DEV_TFRECORD_FILE)
    writeTfRecordData(val_list,input_vocab,label_vocab,VAL_TFRECORD_FILE)




if __name__ == '__main__':
    tfrecord_main()
