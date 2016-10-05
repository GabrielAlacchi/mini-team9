
import tensorflow as tf

import serial
import io

import getopt
import sys

import data_set
import consume_model
import socket


def react_to_label(label):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_addr = ('localhost', 8124)

    try:
        sock.connect(server_addr)
        if label == 'left':
            sock.sendall("backward\n")
        elif label == 'right':
            sock.sendall("forward\n")
        elif label == 'up':
            sock.sendall("toggle-play\n")
    except:
        print "Couldn't connect to the server"
    finally:
        sock.close()


def open_com(port_name, baud):
    return serial.Serial(port_name, baud, timeout=1)


def try_read_line(sio):
    try:
        return sio.readline()
    except:
        return ''


def take_readings(com_port, number_of_readings, model, session):
    sio = io.TextIOWrapper(io.BufferedRWPair(com_port, com_port))

    counter = 0
    lines = []

    print "Arduino is initializing... "

    while "ready" not in try_read_line(sio):
        pass

    print "Arduino is ready!"

    while True:

        while counter < number_of_readings:
            line = try_read_line(sio)
            if 'ANG:' in line:
                lines += [line.strip()]
                counter += 1
            else:
                counter = 0

        print "%d readings taken from %s" % (number_of_readings, com_port.portstr)

        inputs = data_set.fetch_input_from_lines(lines)

        y = session.run(model.inference,
                                   feed_dict={
                                       consume_model.x_pl: inputs
                                   })

        labels = model.labels_from_prediction(y)

        print "Classifier prediction: %s" % labels[0]
        # react_to_label(labels[0])

        lines = []
        counter = 0


def main(argv):
    optlist, argv = getopt.getopt(argv, 'p:n:b:t:', ['--com-port=', '--number-of-reads=', '--baud=', '--train-dir='])

    port = 'ttyUSB0'
    baud = 9600

    train_dir = '.'

    readings = 50

    for opt, arg in optlist:
        if opt in ('-p', '--com-port='):
            port = arg
        elif opt in ('-n', '--number-of-reads='):
            readings = int(arg)
        elif opt in ('-b', '--baud='):
            baud = int(arg)
        elif opt in ('-t', '--train-dir='):
            train_dir = arg

    with tf.Session() as sess:
        print "Loading model..."
        model = consume_model.restore_from_train_sess(sess, train_dir=train_dir)
        print "Model loaded!"

        print "Opening com port."
        com_port = open_com(port, baud)
        take_readings(com_port, readings, model=model, session=sess)

if __name__ == "__main__":
    main(sys.argv[1:])
