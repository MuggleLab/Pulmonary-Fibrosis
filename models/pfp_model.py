# Build Model
from pandas import pd

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D


class PFPModel(Model):
    def __init__(self):
        super().__init__()
        self.build_model()

    def build_model(self):
        self.conv1 = tf.keras.Sequential([
            Conv3D(filters=200, kernel_size=3, padding='same', activation='relu'),
            Conv3D(filters=100, kernel_size=3, padding='same', activation='relu'),
            Conv3D(filters=100, kernel_size=3, padding='same', activation='relu'),
            Conv3D(filters=50, kernel_size=3, padding='same', activation='relu'),
            Flatten(),
        ])

        self.fc = tf.keras.Sequential([
            Dense(500, activation='relu'),
            Dense(100, activation='relu'),
            Dense(1)
        ])

    def call(self, inputs, training=None, mask=None):
        """
        inputs: {imgs: [], info: []}
        """

        imgs = inputs['imgs']
        info = inputs['info']

        imgs = tf.cast(imgs, float)
        info = tf.cast(info, float)

        conv_out = self.conv1(imgs)

        info = tf.concat((conv_out, info), axis=1)
        out = self.fc(info)
        return out


if __name__ == '__main__':
    import os

    root_dir = '~/Data/OSIC'
    train_csv_path = os.path.join(root_dir, 'train.csv')
    test_csv_path = os.path.join(root_dir, 'test.csv')
    img_dir = os.path.join(root_dir, 'preprocessing_data')

    train_data = pd.read_csv(train_csv_path)