
import numpy as np
import model
import math

import os
import re
import transform

import scipy


NOISE_FACTOR = 0.5

# http://stanford.edu/~sxie/Michael%20Xie,%20David%20Pan,%20Accelerometer%20Gesture%20Recognition.pdf
EXP_ALPHA = 0.3
THRESHOLD_ALPHA = 0.22


class DataSet:

    def __init__(self):
        self.batch_index = 0

    def one_hot_to_label(self, one_hot):
        for i in xrange(len(one_hot)):
            if one_hot[i] == 1:
                return self.label_dict[i]

        return None

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


def expand_range(prev_vec, curr_vec):
    delta = curr_vec - prev_vec
    for i in xrange(3):
        if delta[i] < -300.:
            # Wrapped from 180 to -180
            curr_vec[i] += 360.
        elif delta[i] > 300:
            # Wrapped from -180 to 180
            curr_vec[i] -= 360


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
        array[i:i+3] = transform.relative_angles(basis, array[i:i+3])

    # Convert ranges of the angles to -inf to inf to preserve rotation information
    for i in xrange(3, 150, 3):
        prev_vec = array[i-3:i]
        curr_vec = array[i:i+3]
        expand_range(prev_vec=prev_vec, curr_vec=curr_vec)


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


def exp_moving_avg(example_matrix):
    alpha = EXP_ALPHA
    smoothed_matrix = np.zeros([50, 3])
    smoothed_matrix[0] = example_matrix[0]
    for i in xrange(1, 50):
        smoothed_matrix[i] = alpha * example_matrix[i-1] + (1 - alpha) * smoothed_matrix[i-1]

    return smoothed_matrix


def threshold_and_interpolate(matrix):
    mat_rows = matrix.shape[0]

    x_init = matrix[0]
    alpha = THRESHOLD_ALPHA

    norms = np.array(map(lambda x: norm(x - x_init), matrix))
    threshold = alpha * math.sqrt(np.std(norms))

    left = 0
    right = matrix.shape[0]
    # Get the indices to keep
    for ind in xrange(0, matrix.shape[0]):
        # Find the first point above the threshold
        if norms[ind] > threshold:
            left = ind
            break

    # Threshold a bit on the right if we can
    for ind in xrange(matrix.shape[0], matrix.shape[0] - 10, -1):
        # The the threshold at the end
        if norms[ind - 1] > threshold:
            right = ind
            break

    # Slice to cut off noise
    matrix = matrix[left:right]

    # Re interpolate between vectors to get back to 50 points
    num_vectors = matrix.shape[0]

    for ind in xrange(mat_rows - num_vectors):
        # Find an appropriate spot to interpolate between
        inter = int(ind * float(mat_rows) / float(mat_rows - num_vectors)) + 1

        if inter >= num_vectors - 1:
            inter = num_vectors - 2

        # Interpolate
        interpolated_vec = np.sum(matrix[inter:inter+2], axis=0) / 2.
        matrix = np.insert(matrix, inter + 1, values=interpolated_vec, axis=0)

    return matrix


def apply_fft(matrix):
    fft = np.fft.fft2(matrix, s=[25, 3])
    abs_func = np.vectorize(np.absolute)

    fft = abs_func(fft)
    return np.insert(fft, 0, matrix, axis=0)


def map_to_mat(matrix, row_entry):
    for ind in xrange(matrix.shape[0]):
        matrix[ind] = np.copy(row_entry[3*ind:3*ind+3])


def mat_to_row(matrix, row_entry):
    for ind in xrange(0, row_entry.size, 3):
        row_entry[ind:ind+3] = np.copy(matrix[ind/3])

    # Return excess
    if matrix.size > row_entry.size:
        return matrix[row_entry.size / 3:]


def process(data):

    # Stores sample points as row vectors

    fft_entries = np.zeros([data.shape[0], 63])

    for ind in xrange(data.shape[0]):
        # Reshape every given row to a matrix of row vectors
        matrix = np.zeros([50, 3])

        # I don't want to use reshape since it'll disorganize the data
        map_to_mat(matrix, data[ind])
        matrix = exp_moving_avg(example_matrix=matrix)
        matrix = threshold_and_interpolate(matrix)
        matrix = apply_fft(matrix)
        excess = mat_to_row(matrix, data[ind])
        mat_to_row(excess, fft_entries[ind])

    fft_entries = fft_entries.T
    for ind in xrange(fft_entries.shape[0]):
        data = np.insert(data, data.shape[1], fft_entries[ind], axis=1)

    return data


def fetch_data_from_dir(directory):

    print "Loading data from %s" % directory

    sub_dirs = [os.path.join(directory, x) for x in os.listdir(directory) if
                os.path.isdir(os.path.join(directory, x))]

    label_dict = {}
    for j in xrange(len(sub_dirs)):
        label_dict[j] = os.path.basename(sub_dirs[j])

    arrays = []

    num_rows = 0
    for sub_dir in sub_dirs:
        print "Loading data for class %s" % os.path.basename(sub_dir)
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

    print "Done loading data..."

    data_set = DataSet()

    print "Preprocessing data..."
    data_set.inputs = process(data)
    data_set.labels = labels
    data_set.label_dict = label_dict

    print "Data loaded!\n"
    return data_set
