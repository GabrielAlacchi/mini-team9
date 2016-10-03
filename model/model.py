
import math
import tensorflow as tf

NUM_CLASSES = 3
INPUT_SIZE = 150


def hidden_layer(x, input_size, output_size, scope_name):
    # Input enters as a row vector
    with tf.name_scope(scope_name):
        weights = tf.Variable(
            tf.truncated_normal([input_size, output_size], stddev=1.0 / math.sqrt(float(input_size))),
                name=scope_name + '/weights')

        biases = tf.Variable(tf.zeros([output_size]),
                name=scope_name + '/biases')

        hidden = tf.nn.sigmoid(tf.matmul(x, weights) + biases)

    with tf.name_scope(scope_name + '/summaries'):
        tf.histogram_summary(scope_name + '/weights', weights)
        tf.histogram_summary(scope_name + '/biases', biases)

    return hidden


def softmax_classifier(x, input_size, num_classes, scope_name):

    with tf.name_scope(scope_name):
        weights = tf.Variable(
            tf.truncated_normal([input_size, num_classes], stddev=1.0 / math.sqrt(float(input_size))),
            name=scope_name + '/weights')

        biases = tf.Variable(tf.zeros([num_classes]), name=scope_name + '/biases')

        softmax = tf.nn.log_softmax(tf.matmul(x, weights) + biases)

    with tf.name_scope(scope_name + '/summaries'):
        tf.histogram_summary(scope_name + '/weights', weights)
        tf.histogram_summary(scope_name + '/biases', biases)

    return softmax


# Requires that inputs are row vectors of size INPUT_SIZE
def inference(x):

    # hidden1 = hidden_layer(x, INPUT_SIZE, 100, "hidden1")
    # hidden2 = hidden_layer(hidden1, 100, 50, "hidden2")
    softmax = softmax_classifier(x, INPUT_SIZE, NUM_CLASSES, "softmax_linear")

    return softmax


def loss(y, labels):
    labels = tf.cast(labels, dtype=tf.float32)
    cross_entropy = -tf.reduce_sum(labels * y, reduction_indices=[1])

    pre_reg_loss = tf.reduce_mean(cross_entropy, name='loss')

    vars_to_regularize = [v for v in tf.all_variables()
                             if 'weights' in v.name or 'biases' in v.name]

    final_loss = pre_reg_loss
    for tensor in vars_to_regularize:
        final_loss += tf.nn.l2_loss(tensor, name=tensor.op.name + '/l2_regularization')

    return final_loss


def training(loss_function, learning_rate):

    tf.scalar_summary("Loss + L2 Regularization", loss_function)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(loss_function, global_step=global_step)
    return train_op


def evaluate(y, labels):
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))
