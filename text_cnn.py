import tensorflow as tf
import model

class TextCNN(model.Model):
    def __init__(self,
                 hparams,
                 mode,
                 source_vocab_table,
                 target_vocab_table,
                 scope=None,
                 extra_args=None):

        self.filter_sizes = list(map(int, hparams.filter_sizes.split(",")))
        self.num_filters = hparams.num_filters

        super(TextCNN, self).__init__(hparams,
                                      mode,
                                      source_vocab_table,
                                      target_vocab_table,
                                      scope,
                                      extra_args)



    def add_model(self, embedding_inputs):
        """
        :param embedding_inputs: The output of embedding layer: (batch_size, sequence_length, embedded_size)
        :return: The input to the final FC layer, (batch_size, final_fc_size)
        """

        #Expand the input for one more dimension, now it's like (batch_size, sequence_lenth, embeded_size, 1)
        #The last '1' is for the channel number
        with tf.name_scope('CNN'):
            embedded_x = tf.expand_dims(embedding_inputs,-1, name='expand_dims')

            pooled_outputs = []  # to be concatenated
            model_summaries = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope('conv-maxpool-{}'.format(filter_size)):
                    shape = [filter_size, self.embedding_size, 1, self.num_filters]
                    init_W = tf.truncated_normal(shape, stddev=0.1)
                    W = tf.Variable(init_W, name='W')

                    W_summary = tf.summary.histogram('conv-maxpool-{}/W'.format(filter_size), W)
                    model_summaries.append(W_summary)

                    init_b = tf.constant(0.1, shape=[self.num_filters])
                    b = tf.Variable(init_b, name='b')

                    b_summary = tf.summary.histogram('conv-maxpool-{}/b'.format(filter_size), b)
                    model_summaries.append(b_summary)

                    #size of conv is (batch_size, sequence_length-filter_size+1 , 1,num_filters)
                    conv = tf.nn.conv2d(embedded_x, W,
                                        padding="VALID", strides=[1, 1, 1, 1],
                                        name='conv')
                    relu_out = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                    #size of maxpool_out: (batch_size, 1, 1, num_filters)
                    maxpool_out = tf.nn.max_pool(
                        relu_out,
                        ksize=[1, (self.sequence_length - filter_size + 1), 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='pool')
                    pooled_outputs.append(maxpool_out)

            num_filters_total = self.num_filters * len(self.filter_sizes)
            self.model_output_dim = num_filters_total

            # 4d: [batch_size, 1, 1, num_filters]
            # "3" here is tricky
            # so h_pool turns out to be: (batch_size, 1,1, num_filters_total)
            h_pool = tf.concat(pooled_outputs, 3, name='concat')

            # flatten into [batch, num_filters_total]
            h_pool_flat = tf.reshape(h_pool,
                                          [-1, num_filters_total],
                                          name="pooled_outputs")

            with tf.name_scope('dropout'):
                h_dropout = tf.nn.dropout(
                    h_pool_flat, self.dropout_placeholder, name='dropout')

            model_summary = tf.summary.merge(model_summaries)

            return h_dropout, model_summary