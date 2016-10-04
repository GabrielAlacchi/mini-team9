
import tensorflow as tf
from tensorflow.python.platform import gfile
import model
import data_set
import os


def restore_from_train_sess(sess, train_dir):
    print "loading graph..."
    with gfile.FastGFile(os.path.join(train_dir, "graph.pb"), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='model')

    print "done..."


def get_output_op(sess):
    return sess.graph.get_tensor_by_name('softmax_result:0')


def get_eval_op(sess):
    return sess.graph.get_tensor_by_name('eval')


def get_feed_dict(x, labels):

    return {
        'inputs_pl:0': x,
        'labels_pl:0': labels
    }


if __name__ == "__main__":
    with tf.Session() as session:
        restore_from_train_sess(session, './train')
        eval_op = get_eval_op(session)

        x, labels = data_set.fetch_data('./recordings', False)
        feed_dict = get_feed_dict(x, labels)

        print "Accuracy: %.2f" % session.run(eval_op, feed_dict=feed_dict)