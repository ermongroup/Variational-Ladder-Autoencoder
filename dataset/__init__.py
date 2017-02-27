import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from dataset.dataset import *
from dataset.dataset_celeba import CelebADataset
from dataset.dataset_mnist import MnistDataset
from dataset.dataset_svhn import SVHNDataset
from dataset.dataset_lsun import LSUNDataset
