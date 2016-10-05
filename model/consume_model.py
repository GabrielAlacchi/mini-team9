
import tensorflow as tf
import numpy as np

import model
import data_set
import os
import re

x_pl = tf.placeholder(dtype=tf.float32, shape=[None, model.INPUT_SIZE])
labels_pl = tf.placeholder(dtype=tf.float32, shape=[None, model.NUM_CLASSES])


class Model:

    def __init__(self, inference, loss, evaluate, label_dict):
        self.inference = inference
        self.loss = loss
        self.evaluate = evaluate
        self.label_dict = label_dict

    def labels_from_prediction(self, prediction):
        # Returns an array of the index of max for each prediction (row) vector
        predictions = np.argmax(prediction, axis=1)
        # Lookup the label_dict for each argument
        return map(lambda arg: self.label_dict[arg], predictions)


def restore_from_train_sess(sess, train_dir):
    print "loading graph..."
    logits = model.inference(x_pl)

    loss = model.loss(logits, labels_pl)

    eval_op = model.evaluate(logits, labels_pl)

    print "done..."

    saver = tf.train.Saver(tf.all_variables())

    checkpoint_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)
                                                if 'checkpoint' in f
                                                and '.meta' not in f]

    if len(checkpoint_files) == 0:
        raise StandardError("The train directory provided is empty")

    greatest_checkpoint = 0

    for file_path in checkpoint_files:
        items = file_path.split('-')
        if len(items) > 1:
            checkpoint_num = int(items[1])
            greatest_checkpoint = max(checkpoint_num, greatest_checkpoint)

    saver.restore(sess, os.path.join(train_dir, 'checkpoint-%d' % greatest_checkpoint))

    print "Restored session variables..."
    print "Reading labels..."

    label_dict = {}
    with open(os.path.join(train_dir, 'labels.txt'), 'r') as f:
        line = f.readline()
        while line:
            match = re.match("([0-9]+):(.*)", line)
            if match:
                label_dict[int(match.group(1))] = match.group(2)
            line = f.readline()

    return Model(logits, loss, eval_op, label_dict)


if __name__ == "__main__":
    with tf.Session() as session:
        model_graph = restore_from_train_sess(session, './train')

        data = data_set.fetch_data('./recordings', label_dict=model_graph.label_dict)
        x, labels = data.get_all()

        precision = session.run(model_graph.evaluate,
                    feed_dict={
                        x_pl: x,
                        labels_pl: labels
                    })

        print "Precision: %.2f" % precision
