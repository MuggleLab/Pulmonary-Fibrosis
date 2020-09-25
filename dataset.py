import os

import cv2
import pandas as pd
import numpy as np

import tensorflow as tf


class Dataset:
    label_col_name = 'FVC'

    def __init__(self, data_list, patient_list, label_list=None, batch_size=1, root_dir=None):
        self.root_dir = root_dir

        # init variables
        self.data_list = data_list
        self.label_list = label_list if label_list is not None else np.zeros(len(data_list), dtype=float)
        self.patient_list = patient_list

        # init dataset
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (self.data_list, self.label_list, np.arange(len(self.patient_list))))
        self.dataset = self.dataset.map(lambda data, label, index: tf.py_function(self.read_img, [data, label, index],
                                                                                  [tf.float64, tf.float64, tf.float64,
                                                                                   tf.int64]))
        #         self.dataset = self.dataset.repeat(epoch)
        # self.dataset = self.dataset.shuffle(buffer_size=(int(len(data_list) * 0.4) + 3 * batch_size))
        self.dataset = self.dataset.batch(batch_size, drop_remainder=False)

    def __iter__(self):
        return self.dataset.__iter__()

    def read_img(self, data, label, index: tf.Tensor):
        patient = self.patient_list[index]
        #print('Load ', patient)
        img_path = os.path.join(self.root_dir, f'{patient}.npy')
        img = np.multiply(np.load(img_path), -1)
        img = np.expand_dims(img, axis=-1)
        img = np.resize(img, (68, 90, 90, 1))

        return img, data, label, index


if __name__ == '__main__':
    import os

    root_dir = '~/Data/OSIC'
    train_csv_path = os.path.join(root_dir, 'train.csv')
    test_csv_path = os.path.join(root_dir, 'test.csv')
    img_dir = os.path.join(root_dir, 'preprocessing_data')

    train_data = pd.read_csv(train_csv_path)
    dataset = Dataset(train_data)

    for (img, data, label) in dataset.dataset:
        print(img.shape, data.shape, label)
