#!/usr/bin/python
# -*- coding: UTF-8 -*-

from util import load_obj, label_vector_to_index
import argparse
import tensorflow as tf
from train import add_arguments, create_hparams
from text_cnn import TextCNN
import sys
import numpy as np
from tensorflow.python import debug as tf_debug

__author__ = 'Jaycolas'


FLAGS=None
'''
def add_arguments(parser):
    parser.add_argument("--restore_checkpoint", type=str, default="./", help="Checkpoint location of current training")

def create_hparams(flags):
  """Create inferer hparams."""
  return tf.contrib.training.HParams(
      restore_checkpoint = flags.restore_checkpoint
  )'''

def eval(session, model):

    starting_text = "Please paste your email here: "

    while starting_text:
      print ' '.join(generate_email_receiver(session, model, starting_text=starting_text))
      starting_text = raw_input('> ')



def generate_email_receiver(session, model, starting_text='<eos>'):
    """Generate text from the model.
    Args:
        session: tf.Session() object
        model: Object of type RNNLM_Model
        config: A Config() object
        starting_text: Initial text passed to model.
    Returns:
        output: List of word idxs
    """
    # state = model.initial_state.eval()
    # Imagine tokens as a batch size of one, length of len(tokens[0])
    input_vocab = load_obj('input_vocab')
    label_vocab = load_obj('label_vocab')
    tokens = input_vocab.encode_word_list(starting_text.split())
    tokens_length = len(tokens)


    # Pad with zero if token_length is smaller than sequence_lenth, otherwise cut it to sequence_length
    tokens = tokens+[0]*(model.sequence_length-tokens_length) if tokens_length <= model.sequence_length \
                                                              else tokens[0:model.sequence_length]

    #Convert the tokens to np array
    tokens = np.array(tokens)
    #print np.shape(tokens)

    #axis = 0 shall indicate the the batch number, in this evaluation case it is 1
    tokens = np.expand_dims(tokens, axis=0)

    feed={model.input_placeholder:tokens,
          model.dropout_placeholder: 1}

    y_pred = session.run(model.predictions, feed_dict=feed)
    #print np.shape(y_pred)
    #print np.shape(y_pred)
    #y_pred = np.squeeze(y_pred)

    #print ("(VAL) Evaluating the model using val dataset")
    # print prediction
    #y_pred = np.expand_dims(y_pred,axis=0)
    #print np.shape(y_pred)
    _, prediction_result = label_vector_to_index(y_pred, label_vocab)
    #print prediction_result

    #The 1st dimension of prediction_result could be batch_size, in this case since batch size is 1, so returen the 0 index
    prediction_result = prediction_result[0]

    #print np.shape(prediction_result)
    #print prediction_result

    return prediction_result


def infer_receiver(unused_argv):
    hparams = create_hparams(FLAGS)
    input_vocab = load_obj('input_vocab')
    label_vocab = load_obj('label_vocab')

    with tf.Session() as sess:
        #Before training started, need to check if user need to recover any pre-trained model
        cnn = TextCNN(hparams=hparams,
                      mode=tf.contrib.learn.ModeKeys.TRAIN,
                      source_vocab_table=input_vocab,
                      target_vocab_table=label_vocab,
                      scope=None,
                      extra_args=None)

        #tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        chpt = tf.train.latest_checkpoint(hparams.restore_checkpoint)

        if chpt:
            if tf.train.checkpoint_exists(chpt):
                saver.restore(sess, chpt)
                print("Model has been restored from %s"%(hparams.restore_checkpoint))
        else:
            print ("No existing model loaded from %s, exiting"%(hparams.restore_checkpoint))
            return 0

        #if hparams.debug is True:
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        eval(sess, cnn)





if __name__ == '__main__':
    eval_parser = argparse.ArgumentParser()
    add_arguments(eval_parser)
    FLAGS, unparsed = eval_parser.parse_known_args()
    tf.app.run(main=infer_receiver, argv=[sys.argv[0]] + unparsed)



