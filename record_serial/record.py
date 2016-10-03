
import serial
import io

import getopt
import sys

import os.path as path

import numpy as np


def save_file(path, file_content):


def open_com(port_name, baud):
    return serial.Serial(port_name, baud, timeout=1)


def take_readings(com_port, number_of_readings, directory, label_name):
    sio = io.TextIOWrapper(io.BufferedRWPair(com_port, com_port))
    number_of_files = 0
    file_content = ''
    while True:
        for counter in xrange(number_of_readings):
            file_content += com_port.readline()

        file_name = path.join(directory, label_name, 'readings_%d.dat' % number_of_files)
        save_file(path, file_content)
        number_of_files += 1
        file_content = ''

def main(argv):
    optlist, argv = getopt.getopt(argv, 'p:n:b:d:l:', ['--com-port=', '--number-of-reads=', '--baud=', '--save-dir=', '--label-name='])

    port = 'ttyUSB0'
    baud = 9600

    label_name = ''
    directory = ''

    readings = 50

    for opt, arg in optlist:
        if opt in ('-p', '--com-port='):
            port = arg
        elif opt in ('-n', '--number-of-reads='):
            readings = int(arg)
        elif opt in ('-b', '--baud='):
            baud = int(arg)
        elif opt in ('-d', '--save-dir='):
            directory = arg
        elif opt in ('-l', '--label='):
            label_name = arg

    com_port = open_com(port, baud)
    take_readings(com_port, readings, directory, label_name)


if __name__ == "__main__":
    main(sys.argv[1:])