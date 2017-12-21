from dataset import *
import math, os
from glob import glob
import numpy as np
import scipy.misc as misc
from matplotlib import pyplot as plt

class CelebADataset(Dataset):
    def __init__(self, db_path="/data/celebA/img_align", crop=True):
        Dataset.__init__(self)
        self.data_files = glob(os.path.join(db_path, "*.jpg"))
        if len(self.data_files) < 100000:
            print("Only %d images found for celebA, is this right?" % len(self.data_files))
            exit(-1)
        self.train_size = int(math.floor(len(self.data_files) * 0.8))
        self.test_size = len(self.data_files) - self.train_size
        self.train_img = self.data_files[:self.train_size]
        self.test_img = self.data_files[self.train_size:]

        self.train_idx = 0
        self.test_idx = 0
        self.data_dims = [64, 64, 3]

        self.train_cache = np.ndarray((self.train_size, 64, 64, 3), dtype=np.float32)
        self.train_cache_top = 0
        self.test_cache = np.ndarray((self.test_size, 64, 64, 3), dtype=np.float32)
        self.test_cache_top = 0
        self.range = [-1.0, 1.0]
        self.is_crop = crop
        self.name = "celebA"

    """ Return [batch_size, 64, 64, 3] data array """
    def next_batch(self, batch_size):
        # sample_files = self.data[0:batch_size]
        prev_idx = self.train_idx
        self.train_idx += batch_size
        if self.train_idx > self.train_size:
            self.train_idx = batch_size
            prev_idx = 0

        if self.train_idx < self.train_cache_top:
            return self.train_cache[prev_idx:self.train_idx, :, :, :]
        else:
            sample_files = self.train_img[prev_idx:self.train_idx]
            sample = [self.get_image(sample_file, self.is_crop) for sample_file in sample_files]
            sample_images = np.array(sample).astype(np.float32)
            self.train_cache[prev_idx:self.train_idx] = sample_images
            self.train_cache_top = self.train_idx
            return sample_images

    def next_test_batch(self, batch_size):
        prev_idx = self.test_idx
        self.test_idx += batch_size
        if self.test_idx > self.test_size:
            self.test_idx = batch_size
            prev_idx = 0

        if self.test_idx < self.test_cache_top:
            return self.test_cache[prev_idx:self.test_idx, :, :, :]
        else:
            sample_files = self.test_img[prev_idx:self.test_idx]
            sample = [self.get_image(sample_file, self.is_crop) for sample_file in sample_files]
            sample_images = np.array(sample).astype(np.float32)
            self.test_cache[prev_idx:self.test_idx] = sample_images
            self.test_cache_top = self.test_idx
            return sample_images

    def batch_by_index(self, batch_start, batch_end):
        sample_files = self.data_files[batch_start:batch_end]
        sample = [self.get_image(sample_file) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        return sample_images

    @staticmethod
    def get_image(image_path, is_crop=True):
        image = CelebADataset.transform(misc.imread(image_path).astype(np.float), is_crop=is_crop)
        return image

    @staticmethod
    def center_crop(x, crop_h, crop_w=None, resize_w=64):
        if crop_w is None:
            crop_w = crop_h
        h, w = x.shape[:2]
        j = int(round((h - crop_h) / 2.))
        i = int(round((w - crop_w) / 2.))
        return misc.imresize(x[j:j + crop_h, i:i + crop_w],
                                   [resize_w, resize_w])

    @staticmethod
    def full_crop(x):
        if x.shape[0] <= x.shape[1]:
            lb = int((x.shape[1] - x.shape[0]) / 2)
            ub = lb + x.shape[0]
            x = misc.imresize(x[:, lb:ub], [64, 64])
        else:
            lb = int((x.shape[0] - x.shape[1]) / 2)
            ub = lb + x.shape[1]
            x = misc.imresize(x[lb:ub, :], [64, 64])
        return x

    @staticmethod
    def transform(image, npx=108, is_crop=True, resize_w=64):
        # npx : # of pixels width/height of image
        if is_crop:
            cropped_image = CelebADataset.center_crop(image, npx, resize_w=resize_w)
        else:
            cropped_image = CelebADataset.full_crop(image)
        return np.array(cropped_image) / 127.5 - 1.

    """ Transform image to displayable """
    def display(self, image):
        rescaled = np.divide(image + 1.0, 2.0)
        return np.clip(rescaled, 0.0, 1.0)

    def reset(self):
        self.idx = 0


if __name__ == '__main__':
    dataset = CelebADataset()
    while True:
        batch = dataset.next_batch(100)
        print(batch.shape)
        plt.imshow(dataset.display(batch[0]))
        plt.show()