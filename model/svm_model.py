

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
        b = bias_variable([16], name='linear_svm/biases')

        tf.histogram_summary('linear_svm/weights', W)
        tf.histogram_summary('linear_svm/biases', b)

        return tf.add(tf.matmul(x, W), b, name="svm_result")


def loss(y, labels):
    weights = tf.get_variable('linear_svm/weights')

    regularization = 0.5 * tf.reduce_sum(tf.square(W), 'loss/l2')

    hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([None, 1]), 1 - y*labels))

    return tf.add(regularization, SVM_SCALAR*hinge_loss)


def training(loss_function, learning_rate):

    tf.scalar_summary("Hinge Loss + L2 regularization", loss_function)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(loss_function, global_step=global_step)
    return train_op
