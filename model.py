import tensorflow as tf


class Seq2seq(object):
  
  def __init__(self, sess, encoder_vocab_size,
  size, max_input_length, batch_size):
    self.sess = sess
    self.encoder_vocab_size = encoder_vocab_size
    self.size = size
    self.max_input_length = max_input_length
    self.batch_size = batch_size
    self._build_model()



  def _build_model(self):
    encoder_inputs = tf.placeholder(tf.int32, shape=[self.max_input_length, None], name="encoder_input")
    embedding_encoder = tf.get_variable("embedding_encoder", [self.encoder_vocab_size, self.size])
    encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, encoder_inputs)
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(256)

    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp, 
      dtype=tf.float32)
  
  def train(self):
    pass