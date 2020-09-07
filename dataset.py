import os

import cv2
import pandas as pd
import numpy as np

import tensorflow as tf


class Dataset:
    label_col_name = 'FVC'
    root_dir = '/Users/younghun/Data/OSIC/'

    # root_dir = '~/Data/OSIC'

    def __init__(self, data_list, batch_size=10):
        # init labels
        label_list = data_list[self.label_col_name].to_numpy()
        self.patient_list = data_list['Patient'].to_numpy()

        data_list.drop([self.label_col_name, 'Patient'], axis=1, inplace=True)
        data_list = pd.get_dummies(data_list)

        # init dataset
        self.dataset = tf.data.Dataset.from_tensor_slices((data_list, label_list, np.arange(len(self.patient_list))))
        self.dataset = self.dataset.map(lambda data, label, index: tf.py_function(self.read_img, [data, label, index], [tf.float64, tf.float64, tf.int64]))
        # self.dataset = self.dataset.shuffle(buffer_size=(int(len(data_list)) + 1 * batch_size))
        self.dataset = self.dataset.batch(batch_size, drop_remainder=False)

    def __iter__(self):
        # return self.dataset.__iter__()
        for data in self.dataset:
            print(data)

    def read_img(self, data, label, index: tf.Tensor):
        img_path = os.path.join(self.root_dir, 'preprocessing_data', f'{self.patient_list[index]}.npy')
        img = np.load(img_path)
        img.resize((38, 334, 334))

        return img, data, label


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
