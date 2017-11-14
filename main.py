import tensorflow as tf
from model import Seq2seq


tf.app.flags.DEFINE_integer("encoder_vocab_size", 50000,
                          "Encoder vocabrary size.")
tf.app.flags.DEFINE_integer("size", 256,
                          "Size of each model layer.")
tf.app.flags.DEFINE_integer("max_input_length", 200,
                          "Length of max input vector.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                          "Size of training data batch.")
FLAGS = tf.app.flags.FLAGS

def main():
  seq2seq = Seq2seq(tf.Session(),
    FLAGS.encoder_vocab_size,
    FLAGS.size,
    FLAGS.max_input_length,
    FLAGS.batch_size
  )  


if __name__ == '__main__':
  main()