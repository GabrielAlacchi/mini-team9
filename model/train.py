
import tensorflow as tf
import os.path
import model
import time
import data_set

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('training_steps', 25000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 25, 'Batch size.')
flags.DEFINE_string('train_dir', 'train', 'Directory to put training data.')


def train():

    with tf.Graph().as_default():

        x = tf.placeholder(tf.float32, [FLAGS.batch_size, model.INPUT_SIZE])
        labels = tf.placeholder(tf.int32, [model.NUM_CLASSES])

        logits = model.inference(x)

        loss = model.loss(x, labels)

        train_step = model.training(loss, FLAGS.learning_rate)

        eval_correct = model.evaluate(logits, labels)

        summary = tf.merge_all_summaries()

        init = tf.initialize_all_variables()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
            sess.run(init)

            for step in xrange(FLAGS.training_steps):
                start_time = time.time()

                batch = data_set.get_batch(FLAGS.batch_size)

                feed_dict = {
                    x: batch.inputs,
                    labels: batch.labels
                }

                _, loss_value = sess.run([train_step, loss],
                                         feed_dict=feed_dict)

                duration = time.time() - start_time

                if step % 100 == 0:
                    print 'Step %d: loss=%.2f (%.3f sec)' % (step, loss_value, duration)

                if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                    saver.save(sess, checkpoint_file, global_step=step)
                    # Evaluate against the training set.
                    print "\n Evaluation: "
                    precision = sess.run(eval_correct, feed_dict=feed_dict) / FLAGS.batch_size

                    print "Precision %.2f" % precision
