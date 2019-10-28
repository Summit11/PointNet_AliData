import sys
sys.path.append('../')
import os
from provider import file_merge
from hParams import hParams

train_csv_path = hParams.train_csv_path
test_csv_path = hParams.test_csv_path

train_npy_path = hParams.train_npy_path
test_npy_path = hParams.test_npy_path

if not os.path.exists(train_npy_path):
    os.makedirs(train_npy_path)

if not os.path.exists(test_npy_path):
    os.makedirs(test_npy_path)

file_merge(train_csv_path, train_npy_path, is_training=True)
file_merge(test_csv_path, test_npy_path, is_training=False)


