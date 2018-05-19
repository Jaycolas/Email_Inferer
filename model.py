
import tensorflow as tf
import numpy as np
from util import index_to_label_vector, label_vector_to_index
import datetime



class Model(object):
  """Abstracts a Tensorflow graph for a learning task.

  We use various Model classes as usual abstractions to encapsulate tensorflow
  computational graphs. Each algorithm you will construct in this homework will
  inherit from a Model object.
  """
  def __init__(self,
               hparams,
               mode,
               source_vocab_table,
               target_vocab_table,
               scope=None,
               extra_args=None):
    """Create the model.

    Args:
      hparams: Hyperparameter configurations.
      mode: TRAIN | EVAL | INFER
      iterator: Dataset Iterator that feeds data.
      sequence_length: The length (word count) of each sequence
      num_class: The total number of label class
      source_vocab_table: Lookup table mapping source words to ids.
      target_vocab_table: Lookup table mapping target words to ids.
      scope: scope of the model.
      extra_args: model_helper.ExtraArgs, for passing customizable functions.

    """

    self.sequence_length = hparams.sequence_length
    self.num_class = target_vocab_table.vocab_len
    self.embedding_size = hparams.embedding_size
    self.batch_size = hparams.batch_size
    self.input_vocab_size = int(source_vocab_table.vocab_len)
    self.l2_reg_lambda = hparams.l2_reg_lambda
    self.mode = mode
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.accuracy = 0
    self.precision = 0
    self.recall = 0
    self.model_output_dim = 0 #This part will be filled by model itself, and used by the final fc layer

    self.build_graph(hparams)


  def build_graph(self, hparams):
    self.add_placeholder()
    self.embedded_chars, self.embedded_summary = self.add_embedding()
    self.model_output, self.model_summary = self.add_model(self.embedded_chars)
    self.scores, self.predictions, self.l2_loss, self.fc_summary = self.fc_layer(self.model_output)
    self.loss, self.loss_summary = self.add_loss(self.scores, self.l2_loss)
    self.accuracy, self.recall, self.precision, self.perform_summary = self.compute_performance(self.predictions)

    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.learning_rate = tf.constant(hparams.learning_rate)
      # warm-up
      self.learning_rate = self.get_learning_rate_warmup(hparams)
      # decay
      self.learning_rate = self.get_learning_rate_decay(hparams)

      self.lr_summary = tf.summary.scalar('learning_rate', self.learning_rate)

    self.train_op, self.grad_summary = self.add_training_op(self.loss, self.learning_rate)

    self.train_summary = tf.summary.merge([self.embedded_summary, self.model_summary,
                                          self.fc_summary, self.loss_summary, self.lr_summary,
                                          self.perform_summary,self.grad_summary])

    self.dev_summary = tf.summary.merge([self.loss_summary, self.perform_summary])



  def add_placeholder(self):

    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when "None" is in a
    placeholder's shape, it's flexible

    Adds following nodes to the computational graph.
    (When None is in a placeholder's shape, it's flexible)

    input_placeholder: Input placeholder tensor of shape
                       (None, num_steps), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, num_steps), type tf.float32
    dropout_placeholder: Dropout value placeholder (scalar),
                         type tf.float32

    """
    # Placeholders for input, output and dropout
    self.input_placeholder = tf.placeholder(tf.int64, [None, self.sequence_length], name="input_x")
    self.labels_placeholder = tf.placeholder(tf.float32, [None, self.num_class], name="input_y")
    self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout_keep_prob")

  def add_embedding(self):
    """Add embedding layer.

    Hint: This layer should use the input_placeholder to index into the
          embedding.
    Hint: You might find tf.nn.embedding_lookup useful.
    Hint: You might find tf.split, tf.squeeze useful in constructing tensor inputs
    Hint: Check the last slide from the TensorFlow lecture.
    Hint: Here are the dimensions of the variables you will need to create:

      W: (len(self.vocab), embed_size)

    Returns:
      embeded_chars:  a tensor of shape (batch_size, sequence_length, embed_size).
    """
    # The embedding lookup is currently only implemented for the CPU
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
      L = tf.Variable(
        tf.random_uniform([self.input_vocab_size, self.embedding_size], -1.0, 1.0),
        name="L")
      embedded_chars = tf.nn.embedding_lookup(L, self.input_placeholder, name='embedding_lookup')

      embedded_summary = tf.summary.histogram('L', L)

      return embedded_chars, embedded_summary
      #TODO: Figure out why there has to be a -1 expansion
      #self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

  def add_model(self, embedding_inputs):
    """
    :param embedding_inputs: The output of embedding layer: (batch_size, sequence_length, embedded_size)
    :return: The input to the final FC layer, (batch_size, final_fc_size)
    """
    raise NotImplementedError


  def fc_layer(self, model_output):
    """

    :param model_output: The output of model, the input of full connection layer: (batch_size, final_fc_size)
    :return: score and predictions, (batch_size, num_class)
    """
    # Final (unnormalized) scores and predictions
    with tf.name_scope("fc_layer"):
      final_fc_size = self.model_output_dim
      W = tf.get_variable(
        "W",
        shape=[final_fc_size, self.num_class],
        initializer=tf.contrib.layers.xavier_initializer())
      b = tf.Variable(tf.constant(0.1, shape=[self.num_class]), name="b")

      fc_summaries = []
      W_summary = tf.summary.histogram('W',W)
      fc_summaries.append(W_summary)
      b_summary = tf.summary.histogram('b',b)
      fc_summaries.append(b_summary)

      l2_loss = tf.constant(0.0)
      l2_loss += tf.nn.l2_loss(W, name='l2_loss_W')
      l2_loss += tf.nn.l2_loss(b, name='l2_loss_b')

      l2_loss_summary = tf.summary.scalar('l2_loss', l2_loss)
      fc_summaries.append(l2_loss_summary)
      fc_summary = tf.summary.merge(fc_summaries)

      scores = tf.nn.xw_plus_b(model_output, W, b, name="scores")

      #In multi-label classification, we use logistic regression to process the score, and define the predictions
      probas = tf.sigmoid(scores, name='sigmoid_scores')
      predictions = tf.cast(tf.round(probas, name="predictions"), dtype=tf.float32)
      return scores, predictions, l2_loss, fc_summary

  def add_loss(self, fc_layer_output, l2_loss):
    """

    :param fc_layer_output: (batch_size, num_class)
    :return: mean cross entrophy loss over the whole batch
    """

    # Calculate mean cross-entropy loss
    with tf.name_scope("loss"):
      losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=fc_layer_output, labels=self.labels_placeholder, name='cross_entropy')
      loss = tf.reduce_mean(losses, name='reduce_mean') + self.l2_reg_lambda * l2_loss
      loss_summary = tf.summary.scalar('loss', loss)

    return loss, loss_summary

      # Accuracy

  def compute_performance(self, predictions):
    with tf.name_scope("accuracy"):
      correct_predictions = tf.equal(predictions, self.labels_placeholder)
      accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    with tf.name_scope('performance'):
      precision = tf.metrics.precision(self.labels_placeholder, predictions, name="precision-micro")[1]
      recall = tf.metrics.recall(self.labels_placeholder, predictions, name="recall-micro")[1]

      prec_summary = tf.summary.scalar("precision", precision)
      rec_summary = tf.summary.scalar("recall", recall)
      perform_summary = tf.summary.merge([prec_summary, rec_summary])

    return accuracy, recall, precision, perform_summary

  def add_training_op(self, loss, learning_rate):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See

    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

    for more information.

    Hint: Use tf.train.AdamOptimizer for this model.
          Calling optimizer.minimize() will return a train_op object.

    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """



    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    # summary operation
    grad_summaries = []
    for grad, v in grads_and_vars:
      if grad is not None:
        hist = tf.summary.histogram("{}/grad/hist".format(v.name), grad)
        sparsity = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(grad))
        grad_summaries.append(hist)
        grad_summaries.append(sparsity)
    grad_summary = tf.summary.merge(grad_summaries)

    return train_op, grad_summary

  def get_learning_rate_warmup(self, hparams):
    """Get learning rate warmup."""
    warmup_steps = hparams.warmup_step
    warmup_scheme = hparams.warmup_scheme
    print("learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
                    (hparams.learning_rate, warmup_steps, warmup_scheme))

    # Apply inverse decay if global steps less than warmup steps.
    # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
    # When step < warmup_steps,
    #   learing_rate *= warmup_factor ** (warmup_steps - step)
    if warmup_scheme == "t2t":
      # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
      warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
      inv_decay = warmup_factor**(
          tf.to_float(warmup_steps - self.global_step))
    elif not warmup_scheme:
      inv_decay = 1
    else:
      raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

    return tf.cond(
        self.global_step < hparams.warmup_step,
        lambda: inv_decay * self.learning_rate,
        lambda: self.learning_rate,
        name="learning_rate_warump_cond")

  def get_learning_rate_decay(self, hparams):
      """Get learning rate decay."""
      if hparams.decay_scheme == "luong10":
        start_decay_step = int(hparams.num_train_steps / 2)
        remain_steps = hparams.num_train_steps - start_decay_step
        decay_steps = int(remain_steps / 10)  # decay 10 times
        decay_factor = 0.5
      elif hparams.decay_scheme == "luong234":
        start_decay_step = int(hparams.num_train_steps * 2 / 3)
        remain_steps = hparams.num_train_steps - start_decay_step
        decay_steps = int(remain_steps / 4)  # decay 4 times
        decay_factor = 0.5
      elif not hparams.decay_scheme:  # no decay
        start_decay_step = hparams.num_train_steps
        decay_steps = 0
        decay_factor = 1.0
      else:
        raise ValueError("Unknown decay scheme %s" % hparams.decay_scheme)

      print("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
                      "decay_factor %g" % (hparams.decay_scheme,
                                           start_decay_step,
                                           decay_steps,
                                           decay_factor))

      return tf.cond(
        self.global_step < start_decay_step,
        lambda: self.learning_rate,
        lambda: tf.train.exponential_decay(
          self.learning_rate,
          (self.global_step - start_decay_step),
          decay_steps, decay_factor, staircase=True),
        name="learning_rate_decay_cond")

  def train_step(self, sess, train_x, train_y, summary_writer, label_vocab, hparams):
    start_time = datetime.datetime.now()
    out_y, out_x = sess.run([train_y, train_x])
    label_vector = index_to_label_vector(out_y, label_vocab)

    feed_dict = {self.input_placeholder: out_x,
                 self.labels_placeholder: label_vector,
                 self.dropout_placeholder: hparams.dropout_keep_prob}

    _, current_step, learning_rate, summaries, accuracy, loss, prec, rec = sess.run(
      [self.train_op, self.global_step, self.learning_rate,  self.train_summary, self.accuracy, self.loss, self.precision, self.recall],
      feed_dict=feed_dict)


    summary_writer.add_summary(summaries, current_step)

    end_time = datetime.datetime.now()
    dur = (end_time - start_time).seconds

    print("(TRAIN) at step {}: loss={:.4f}, learning_rate = {:.4f}, accuracy={:4f}, precision={:4f}, recall={:4f}, cost_time = {}s".format(current_step, loss,
                                                                                  learning_rate, accuracy, prec, rec, dur))


  def dev_step(self, sess, dev_x, dev_y, summary_writer, label_vocab):
    out_y, out_x = sess.run([dev_y, dev_x])
    label_vector = index_to_label_vector(out_y, label_vocab)

    feed_dict = {self.input_placeholder: out_x,
                 self.labels_placeholder: label_vector,
                 self.dropout_placeholder: 1.0}

    current_step, summaries, accuracy, loss, prec, rec = sess.run(
      [self.global_step, self.dev_summary, self.accuracy, self.loss, self.precision, self.recall],
      feed_dict=feed_dict)

    print("(DEV) after step {}: loss={:.2f}, accuracy={:4f}, precision={:4f}, recall={:4f}".format(current_step, loss,
                                                                                  accuracy, prec, rec))
    summary_writer.add_summary(summaries, current_step)

  def eval_step(self, sess, val_x, val_y, summary_writer, input_vocab, label_vocab):
    out_y, out_x = sess.run([val_y, val_x])

    if np.shape(out_y)[0]<self.batch_size:
      print "Not enough samples in this batch"
      return

    feed_dict = {self.input_placeholder: out_x,
                 #self.labels_placeholder: None,
                 self.dropout_placeholder: 1.0}

    prediction = sess.run([self.predictions], feed_dict=feed_dict)

    print ("(VAL) Evaluating the model using val dataset")
    #print prediction
    print out_y
    prediction_index, prediction_result = label_vector_to_index(prediction, label_vocab)
    real_result = map(lambda x:label_vocab.decode_index_list(x), out_y)

    for predict, real in zip(prediction_result, real_result):
      print "Real result is ", real
      print "Predicted result is", predict





