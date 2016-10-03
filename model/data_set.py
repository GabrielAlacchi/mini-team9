
import numpy as np
import model
import math

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


def fetch_data(dir, test):
    if test:
        return fetch_test_data()

    return None


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
