from dataset import *
import os
from matplotlib import pyplot as plt
import numpy as np

class LSUNDataset(Dataset):
    def __init__(self, db_path="/data/data/lsun/bedroom"):
        Dataset.__init__(self)
        self.data_dims = [64, 64, 3]
        self.name = "lsun"

        self.db_path = db_path
        self.db_files = os.listdir(self.db_path)
        self.cur_batch_ptr = 0
        self.cur_batch = self.load_new_data()
        self.train_batch_ptr = 0
        self.train_size = len(self.db_files) * 10000
        self.test_size = self.train_size
        self.range = [-1.0, 1.0]

    def load_new_data(self):
        filename = os.path.join(self.db_path, self.db_files[self.cur_batch_ptr])
        self.cur_batch_ptr += 1
        if self.cur_batch_ptr == len(self.db_files):
            self.cur_batch_ptr = 0
        return np.load(filename) * 2.0 - 1.0

    def next_batch(self, batch_size):
        prev_batch_ptr = self.train_batch_ptr
        self.train_batch_ptr += batch_size
        if self.train_batch_ptr > self.cur_batch.shape[0]:
            self.train_batch_ptr = batch_size
            prev_batch_ptr = 0
            self.cur_batch = self.load_new_data()
        return self.cur_batch[prev_batch_ptr:self.train_batch_ptr, :, :, :]

    def next_test_batch(self, batch_size):
        return self.next_batch(batch_size)

    """ Transform image to displayable """
    def display(self, image):
        rescaled = np.divide(image + 1.0, 2.0)
        return np.clip(rescaled, 0.0, 1.0)


if __name__ == '__main__':
    dataset = LSUNDataset()
    while True:
        batch = dataset.next_batch()
        plt.imshow(dataset.display(batch[0]))
        plt.show()
        for i in range(99):
            dataset.next_batch()
