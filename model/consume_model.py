
import tensorflow as tf
from tensorflow.python.platform import gfile
import model
import data_set
import os

x_pl = tf.placeholder(dtype=tf.float32, shape=[None, model.INPUT_SIZE])
labels_pl = tf.placeholder(dtype=tf.float32, shape=[None, model.NUM_CLASSES])


class Model:

    def __init__(self, inference, loss, evaluate):
        self.inference = inference
        self.loss = loss
        self.evaluate = evaluate


def restore_from_train_sess(sess, train_dir):
    print "loading graph..."
    logits = model.inference(x_pl)

    loss = model.loss(logits, labels_pl)

    eval_op = model.evaluate(logits, labels_pl)

    print "done..."

    saver = tf.train.Saver(tf.all_variables())
    saver.restore(sess, os.path.join('train', 'checkpoint-19999'))

    return Model(logits, loss, eval_op)


if __name__ == "__main__":
    with tf.Session() as session:
        model_graph = restore_from_train_sess(session, './train')

        data = data_set.fetch_data('./alternate_test', False)
        x, labels = data.get_all()

        precision = session.run(model_graph.evaluate,
                    feed_dict={
                        x_pl: x,
                        labels_pl: labels
                    })

        print "Precision: %.2f" % precision
