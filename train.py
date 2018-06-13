#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function
import tensorflow as tf
import os
from text_cnn import  TextCNN
from util import load_obj, save_obj,index_to_label_vector
import time
from tensorflow.python import debug as tf_debug
import datetime
import argparse
import sys



__author__ = 'Jaycolas'

'''
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# model parameters
tf.flags.DEFINE_integer("embedding_size", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 50, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("sequence_length", 200, "Unified sequence length for each email (default:400)")
tf.flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate (default:0.01)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.01, "l2 regularization lambda value (default:0.01)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 20, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every_epoch", 60, "Save model after this many epoch (default: 100)")
tf.flags.DEFINE_string("restore_checkpoint", "./", "Checkpoint location of current training")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("warmup_scheme", "", "The scheme for learning rate warm up (default: t2t)")
tf.flags.DEFINE_string("decay_scheme", "", "The scheme for learning rate decay (default: luong10)")
tf.flags.DEFINE_integer("warmup_step", "10", "The global step when learning rate warm up begins (default: 10)")
tf.flags.DEFINE_integer("num_train_steps", 300, "The maximum total number of train steps")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("debug", False, "Enable debug or not(default: False)")


ROOT_PATH ='./dataset/email'
TRAIN_TFRECORD_DATA = os.path.join(ROOT_PATH, 'train.tfrecords')
DEV_TFRECORD_DATA   = os.path.join(ROOT_PATH, 'dev.tfrecords')
VAL_TFRECORD_DATA   = os.path.join(ROOT_PATH, 'val.tfrecords')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
'''


ROOT_PATH ='./dataset/email'
TRAIN_TFRECORD_DATA = os.path.join(ROOT_PATH, 'train.tfrecords')
DEV_TFRECORD_DATA   = os.path.join(ROOT_PATH, 'dev.tfrecords')
VAL_TFRECORD_DATA   = os.path.join(ROOT_PATH, 'val.tfrecords')

FLAGS=None

def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # Data related
    parser.add_argument("--dev_sample_percentage", type=float, default=0.1, help="Percentage of the training data to use for validation")

    # Model parameters
    parser.add_argument("--embedding_size", type=int, default= 128, help="Dimensionality of character embedding (default: 128)")
    parser.add_argument("--filter_sizes", type=str, default="3,4,5", help="Comma-separated filter sizes (default: '3,4,5')")
    parser.add_argument("--num_filters", type=int, default=50, help="Number of filters per filter size (default: 128)")
    parser.add_argument("--dropout_keep_prob", type=float, default=0.8, help="Dropout keep probability (default: 0.5)")
    parser.add_argument("--sequence_length", type=int, default=200, help="Unified sequence length for each email (default:400)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Initial learning rate (default:0.01)")
    parser.add_argument("--l2_reg_lambda", type=float, default=0.01, help="l2 regularization lambda value (default:0.01)")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch Size (default: 64)")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs (default: 200)")
    parser.add_argument("--evaluate_every", type=int, default=20, help="Evaluate model on dev set after this many steps (default: 100)")
    parser.add_argument("--checkpoint_every_epoch", type=int, default=60, help="Save model after this many epoch (default: 100)")
    parser.add_argument("--restore_checkpoint", type=str, default="./", help="Checkpoint location of current training")
    parser.add_argument("--num_checkpoints", type=int, default=5, help="Number of checkpoints to store (default: 5)")
    parser.add_argument("--warmup_scheme", type=str,default="", help="The scheme for learning rate warm up (default: t2t)")
    parser.add_argument("--decay_scheme", type=str, default="", help="The scheme for learning rate decay (default: luong10)")
    parser.add_argument("--warmup_step", type=int,default="10", help="The global step when learning rate warm up begins (default: 10)")
    parser.add_argument("--num_train_steps", type=int, default=300, help="The maximum total number of train steps")
    parser.add_argument("--num_gpus", type=int, default=0, help="GPUs you wana use in your training, default is 0, will apply model on CPU")

    # Misc Parameters
    parser.add_argument("--allow_soft_placement", type=bool, default=True, help="Allow device soft device placement")
    parser.add_argument("--log_device_placement", type=bool, default=False, help="Log placement of ops on devices")
    parser.add_argument("--debug", type=bool,default=False, help="Enable debug or not(default: False)")


def _parse_function(example_proto):
  features = {"y": tf.VarLenFeature(tf.int64),
              "x": tf.VarLenFeature(tf.int64)}
  parsed_features = tf.parse_single_example(example_proto, features)
  return tf.sparse_tensor_to_dense(parsed_features["y"]), tf.sparse_tensor_to_dense(parsed_features["x"])


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
      debug=flags.debug
  )


def train(unused_argv):

    #1. Preparing dataset
    hparams = create_hparams(FLAGS)
    train_dataset = tf.data.TFRecordDataset(TRAIN_TFRECORD_DATA)
    dev_dataset = tf.data.TFRecordDataset(DEV_TFRECORD_DATA)
    val_dataset = tf.data.TFRecordDataset(VAL_TFRECORD_DATA)
    input_vocab = load_obj('input_vocab')
    label_vocab = load_obj('label_vocab')
    max_length = hparams.sequence_length
    batch_size = hparams.batch_size
    train_dataset = train_dataset.map(_parse_function)
    dev_dataset = dev_dataset.map(_parse_function)
    val_dataset = val_dataset.map(_parse_function)
    #Filter some very long email, we could consider to utilize it in the future
    train_dataset = train_dataset.filter(lambda y,x:tf.less(tf.shape(x)[0],max_length))
    dev_dataset = dev_dataset.filter(lambda y,x:tf.less(tf.shape(x)[0],max_length))
    val_dataset = val_dataset.filter(lambda y,x:tf.less(tf.shape(x)[0],max_length))


    #train_dataset = train_dataset.repeat(hparams.num_epochs) #repeat the dataset indefinitely
    train_dataset  = train_dataset.padded_batch(batch_size, padded_shapes=([None],[max_length]))
    dev_dataset   = dev_dataset.padded_batch(batch_size, padded_shapes=([None],[max_length]))
    val_dataset   = val_dataset.padded_batch(batch_size, padded_shapes=([None],[max_length]))


    train_iterator = train_dataset.make_initializable_iterator()
    train_y,train_x = train_iterator.get_next()

    dev_iterator = dev_dataset.make_initializable_iterator()
    dev_y, dev_x = dev_iterator.get_next()

    val_iterator = val_dataset.make_initializable_iterator()
    val_y, val_x = val_iterator.get_next()

    #2.

    with tf.Session(config=tf.ConfigProto(log_device_placement=hparams.log_device_placement)) as sess:
        cnn = TextCNN(hparams=hparams,
                      mode=tf.contrib.learn.ModeKeys.TRAIN,
                      source_vocab_table=input_vocab,
                      target_vocab_table=label_vocab,
                      scope = None,
                      extra_args = None)


        # IO direction stuff
        #timestamp = str(int(time.time()))
        timestamp = time.strftime("%Y-%m-%d %X", time.localtime())
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

        # summary writer
        train_summary_dir = os.path.join(out_dir, "summary/train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        dev_summary_dir = os.path.join(out_dir, "summary/dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        val_summary_dir = os.path.join(out_dir, "summary/val")
        val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

        # checkpoint writer
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # summary operation
        '''
        grad_summaries = []
        for grad, v in cnn.grads_and_vars:
            if grad is not None:
                hist = tf.summary.histogram("{}/grad/hist".format(v.name), grad)
                sparsity = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(grad))
                grad_summaries.append(hist)
                grad_summaries.append(sparsity)
        grad_summary = tf.summary.merge(grad_summaries)

        prec_summary = tf.summary.scalar("precision", cnn.precision)
        rec_summary = tf.summary.scalar("recall", cnn.recall)
        loss_summary = tf.summary.scalar("loss", cnn.loss)

        train_summary_op = tf.summary.merge([grad_summary, prec_summary, rec_summary, loss_summary])
        dev_summary_op = tf.summary.merge([prec_summary, rec_summary, loss_summary])
        '''

        # real code

        #Before training started, need to check if user need to recover any pre-trained model
        chpt = tf.train.latest_checkpoint(hparams.restore_checkpoint)
        print(chpt)
        if chpt:
            if tf.train.checkpoint_exists(chpt):
                saver.restore(sess, chpt)
                print("Model has been resotre from %s"%(hparams.restore_checkpoint))
        else:
            sess.run(tf.global_variables_initializer())
            print("No pre-trained model loaeded, initialized all variable")

        #Local variables and iterator could not be saved, we need to initialize them again
        sess.run(tf.local_variables_initializer())
        sess.run([train_iterator.initializer, dev_iterator.initializer, val_iterator.initializer], feed_dict=None)

        #Only for debug purpose
        if hparams.debug is True:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        for epoch in range(hparams.num_epochs):
            print("Starting epoch %d"%(epoch))
            epoch_start_time = datetime.datetime.now()

            while True:
                try:
                    cnn.train_step(sess, train_x, train_y, train_summary_writer, label_vocab, hparams)

                except tf.errors.OutOfRangeError:
                    #One epoch has been ended, we need to re-initiate dev dataset
                    sess.run(train_iterator.initializer, feed_dict=None)
                    break

            epoch_end_time = datetime.datetime.now()
            epoch_dur = (epoch_end_time-epoch_start_time).seconds

            print("epoch costed %d"%(epoch_dur))

            # Run the Dev dataset for every epoch
            if epoch % hparams.evaluate_every == 0:
                while True:
                    try:
                        cnn.dev_step(sess, dev_x, dev_y, dev_summary_writer, label_vocab)
                    except tf.errors.OutOfRangeError:
                        sess.run(dev_iterator.initializer, feed_dict=None)
                        break

            if epoch % hparams.checkpoint_every_epoch == 0:
                saver.save(sess, checkpoint_prefix, global_step=cnn.global_step)
                print("Saved model checkpoint to {}\n".format(checkpoint_prefix))
                #Evaluate the model very time when saving it
                while True:
                    try:
                        cnn.eval_step(sess, val_x, val_y, val_summary_writer, input_vocab, label_vocab)
                    except tf.errors.OutOfRangeError:
                        sess.run(val_iterator.initializer, feed_dict=None)
                        break


if __name__ == "__main__":
  nmt_parser = argparse.ArgumentParser()
  add_arguments(nmt_parser)
  FLAGS, unparsed = nmt_parser.parse_known_args()
  tf.app.run(main=train, argv=[sys.argv[0]] + unparsed)






