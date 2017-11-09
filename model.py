import tensorflow as tf


class Seq2seq(object):
  
  def __init__(self, sess):
    self.sess = sess
    self._build_model()



  def _build_model(self, encoder_inputs):
   
    embedding_encoder = tf.variable_scope.get_variable(
        "embedding_encoder", [src_vocab_size, embedding_size]
    encoder_emb_inp = tf.embedding_ops.embedding_lookup(
        embedding_encoder, encoder_inputs)

  
  def train(self):
    pass