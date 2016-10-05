
import tensorflow as tf
import os.path
import model
import time
import data_set

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('training_steps', 20000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 45, 'Batch size.')
flags.DEFINE_string('train_dir', 'train', 'Directory to put training data.')
flags.DEFINE_bool('test_data', False, 'Whether or not to use test_data')


def train():

    with tf.Graph().as_default():

        data = data_set.fetch_data('./recordings', FLAGS.test_data)

        x = tf.placeholder(tf.float32, [None, model.INPUT_SIZE], name='inputs_pl')
        labels = tf.placeholder(tf.int32, [None, model.NUM_CLASSES], name='labels_pl')

        logits = model.inference(x)

        loss = model.loss(logits, labels)

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

                inp, lab = data.get_batch(FLAGS.batch_size)

                feed_dict = {
                    x: inp,
                    labels: lab
                }

                _, loss_value = sess.run([train_step, loss],
                                         feed_dict=feed_dict)

                duration = time.time() - start_time

                if step % 100 == 0:
                    print 'Step %d: loss=%.2f (%.3f sec)' % (step, loss_value, duration)
                    # Update the events file.
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.training_steps:
                    checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                    saver.save(sess, checkpoint_file, global_step=step)
                    # Evaluate against the training set.
                    print "\n Evaluation: "
                    val_inputs, val_labels = data.get_all()
                    precision = sess.run(eval_correct, feed_dict={
                        x: val_inputs,
                        labels: val_labels
                    })

                    print "Precision %.2f" % precision

            # Validation
            validation_data = data_set.fetch_data('./recordings', False)
            test_gestures, test_labels = validation_data.get_all()
            feed_dict = {
                x: test_gestures,
                labels: test_labels
            }
            accuracy_on_test = sess.run(eval_correct, feed_dict=feed_dict)
            print "Final accuracy: %.2f" % accuracy_on_test

            print "Exporting graph and parameters..."
            tf.train.write_graph(sess.graph_def, FLAGS.train_dir, "graph.pb", False)  # proto
            final_saver = tf.train.Saver(tf.all_variables())
            final_saver.save(sess, os.path.join(FLAGS.train_dir, "graph_params.data"))

            print "Exporting labels..."
            label_dict = validation_data.label_dict

            with open(os.path.join(FLAGS.train_dir, "labels.txt"), 'w') as f:
                for key, val in label_dict.iteritems():
                    f.write("%d:%s\n" % (key, val))


if __name__ == "__main__":
    train()
