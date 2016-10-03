
import serial
import io

import getopt
import sys

import os.path as path
import os


def save_file(file_path, file_content):
    directory = path.dirname(file_path)

    if not path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'w') as file_stream:
        file_stream.write(file_content)


def open_com(port_name, baud):
    return serial.Serial(port_name, baud, timeout=1)


def try_read_line(sio):
    try:
        return sio.readline()
    except:
        return ''


def take_readings(com_port, number_of_readings, directory, label_name):
    sio = io.TextIOWrapper(io.BufferedRWPair(com_port, com_port))
    number_of_files = 0
    counter = 0
    file_content = ''
    while True:
        while counter < number_of_readings:
            line = try_read_line(sio)
            if 'ANG:' in line:
                file_content += line
                counter += 1

        print "%d readings taken from %s" % (number_of_readings, com_port.portstr)

        file_name = path.join(directory, label_name, 'readings_%d.dat' % number_of_files)
        save_file(file_name, file_content)
        print "Readings written to file: %s" % file_name

        number_of_files += 1
        print "%d files written" % number_of_files

        file_content = ''
        counter = 0


def main(argv):
    optlist, argv = getopt.getopt(argv, 'p:n:b:d:l:', ['--com-port=', '--number-of-reads=', '--baud=', '--save-dir=', '--label-name='])

    port = 'ttyUSB0'
    baud = 9600

    label_name = ''
    directory = '.'

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

    if label_name == '':
        print "Missing argument label_name, -l"
        exit(2)

    com_port = open_com(port, baud)
    take_readings(com_port, readings, directory, label_name)


if __name__ == "__main__":
    main(sys.argv[1:])
