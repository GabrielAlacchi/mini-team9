

import tensorflow as tf

NUM_CLASSES = 3
INPUT_SIZE = 213

SVM_SCALAR = 1.


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def inference(x):

    with tf.name_scope('linear_svm'):
        W = weight_variable([INPUT_SIZE, NUM_CLASSES], name='linear_svm/weights')
        b = bias_variable([NUM_CLASSES], name='linear_svm/biases')

        tf.histogram_summary('linear_svm/weights', W)
        tf.histogram_summary('linear_svm/biases', b)

        return tf.add(tf.matmul(x, W), b, name="svm_result")


def loss(y, labels):
    labels = tf.cast(labels, dtype=tf.float32)

    weights = [v for v in tf.all_variables() if 'linear_svm/weights' in v.name]

    regularization = 0.5 * tf.reduce_sum(tf.square(weights[0]), name='l2_loss')

    hinge_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.relu(1. - y*labels), reduction_indices=[1]), name="hinge_loss")

    svm_loss = tf.add(regularization, SVM_SCALAR*hinge_loss, name='svm_loss')

    tf.scalar_summary('L2 Loss', regularization)
    tf.scalar_summary('Hinge Loss', hinge_loss)
    tf.scalar_summary('SVM Loss', svm_loss)

    return svm_loss


def training(loss_function, learning_rate):

    tf.scalar_summary("Hinge Loss + L2 regularization", loss_function)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(loss_function, global_step=global_step)
    return train_op


def evaluate(y, labels):
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32), name='eval')
