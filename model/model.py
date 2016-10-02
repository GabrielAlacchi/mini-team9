
import math
import tensorflow as tf

NUM_CLASSES = 3
INPUT_SIZE = 150


def hidden_layer(x, input_size, output_size, scope_name):
    # Input enters as a row vector
    with tf.name_scope(scope_name):
        weights = tf.Variable(
            tf.truncated_normal([input_size, output_size], stddev=1.0 / math.sqrt(float(input_size))), name='weights')

        biases = tf.Variable(tf.zeros([output_size]), name='biases')

        hidden = tf.nn.relu(tf.matmul(x, weights) + biases)

    with tf.name_scope(scope_name + '/summaries'):
        tf.histogram_summary('weights', weights)
        tf.histogram_summary('biases', biases)

    return hidden


def softmax_classifier(x, input_size, num_classes, scope_name):

    with tf.name_scope(scope_name):
        weights = tf.Variable(
            tf.truncated_normal([input_size, num_classes], stddev=1.0 / math.sqrt(float(input_size))), name='weights')

        biases = tf.Variable(tf.zeros([num_classes]), name='biases')

        softmax = tf.nn.softmax(tf.matmul(x, weights) + biases)

    with tf.name_scope(scope_name + '/summaries'):
        tf.histogram_summary('weights', weights)
        tf.histogram_summary('biases', biases)

    return softmax


# Requires that inputs are row vectors of size INPUT_SIZE
def inference(x):

    hidden = hidden_layer(x, INPUT_SIZE, 50, "hidden")
    softmax = softmax_classifier(hidden, 50, NUM_CLASSES, "softmax_linear")

    return softmax


def loss(y, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, labels, name='cross_entropy')
    mean_loss = tf.reduce_mean(cross_entropy, name='loss')

    return mean_loss


def training(loss_function, learning_rate):

    tf.scalar_summary(loss_function.op.name, loss_function)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluate(y, labels):
    correct = tf.nn.in_top_k(y, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
