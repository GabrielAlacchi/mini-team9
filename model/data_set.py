
import numpy as np
import model
import math

import os
import re

NOISE_FACTOR = 0.5


class DataSet:

    def __init__(self):
        self.batch_index = 0

    def get_all(self):
        return self.inputs, self.labels

    def get_batch(self, batch_size):
        num_left = self.inputs.shape[0] - self.batch_index
        if num_left < batch_size:
            self.batch_index = 0

        range_min, range_max = self.batch_index, self.batch_index + batch_size
        inputs = self.inputs[range_min:range_max]
        labels = self.labels[range_min:range_max]

        self.batch_index += batch_size

        return inputs, labels


def fetch_data(directory, test):
    if test:
        return fetch_test_data()
    else:
        return fetch_data_from_dir(directory)


def norm(vector):
    return math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)


def init_input_array(array, basis_vec):

    # Without this noise there are literally 3 combinations to optimize around
    # so the model converges unrealistically fast
    noise_vec = basis_vec + NOISE_FACTOR * np.random.rand(3)

    # Varies the angular velocity over measurements
    growth_rate = np.random.rand(1)[0] * NOISE_FACTOR
    # Put data relative to the first vector
    for i in xrange(0, 150, 3):
        array[i:i+3] = i * (1 - math.exp(-i * growth_rate)) * norm(basis_vec) * noise_vec


def fetch_test_data():
    # Fetch 150 records for each of the 3 class labels
    data = DataSet()

    row_count = model.NUM_CLASSES * 500
    inputs = np.random.rand(row_count, model.INPUT_SIZE)
    labels = np.zeros([row_count, model.NUM_CLASSES], dtype=np.int32)

    basis = np.random.rand(3, 3)  # np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0]])

    # Initializes the input vectors
    for i in xrange(row_count):
        init_input_array(inputs[i], basis[i % model.NUM_CLASSES])
        labels[i][i % model.NUM_CLASSES] = 1

    data.inputs = inputs
    data.labels = labels
    return data


def write_array(lines, array):
    i = 0
    for line in lines:
        match = re.match("!ANG:([^,]+),([^,]+),([^,]+)", line)
        if match and i < array.shape[0]:
            array[i:i+3] = [match.group(1), match.group(2), match.group(3)]
            i += 3

    # Make all entries relative to the first
    basis = np.copy(array[:3])
    for i in xrange(0, 150, 3):
        array[i:i+3] = array[i:i+3] - basis


def fetch_all_files(sub_dir):
    files = [os.path.join(sub_dir, x) for x in os.listdir(sub_dir)]

    array = np.zeros([len(files), 150])
    row = 0
    for f in files:
        with open(f, 'r') as file_stream:
            lines = file_stream.readlines()
            write_array(lines, array[row])
            row += 1

    return array


def fetch_data_from_dir(directory):
    sub_dirs = [os.path.join(directory, x) for x in os.listdir(directory) if
                os.path.isdir(os.path.join(directory, x))]

    num_classes = len(sub_dirs)
    arrays = []

    num_rows = 0
    for sub_dir in sub_dirs:
        label_name = os.path.basename(sub_dir)
        array = fetch_all_files(sub_dir)
        arrays += [array]
        num_rows += array.shape[0]

    data = np.zeros([num_rows, 150])
    labels = np.zeros([num_rows, 3])

    for i in xrange(num_rows):
        array = arrays[i % len(arrays)]
        index = i / 3
        data[i] = array[index]
        labels[i][i % len(arrays)] = 1

    data_set = DataSet()
    data_set.inputs = data
    data_set.labels = labels
    return data_set
