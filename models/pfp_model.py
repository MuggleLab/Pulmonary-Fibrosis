# Build Model
import sys

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, AveragePooling3D
from tensorflow.python.keras.layers import Lambda
from tensorflow.keras.backend import mean


class PFPModel(Model):
    def __init__(self):
        super().__init__()
        self.C1, self.C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")

        self.build_model()

    def build_model(self):
        #data_format = 'channels_first'
        self.conv1 = tf.keras.Sequential([
            Conv3D(filters=10, kernel_size=5, padding='same', activation='relu'),
            Conv3D(filters=5, kernel_size=5, padding='same', activation='relu'),
            Conv3D(filters=5, kernel_size=5, strides=2, padding='valid', activation='relu'),
            # AveragePooling3D(pool_size=10),
            Conv3D(filters=1, kernel_size=5, strides=2, padding='valid', activation='relu'),
            # AveragePooling3D(pool_size=10),
        ])
        self.flat = Flatten()
        self.fc = tf.keras.Sequential([
            Dense(100, activation='relu'),
            Dense(100, activation='relu'),
        ])

        self.p1 = Dense(3, activation="linear", name="p1")
        self.p2 = Dense(3, activation="relu", name="p2")
        self.preds = Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds")

    def score(self, y_true, y_pred):
        tf.dtypes.cast(y_true, tf.float32)
        tf.dtypes.cast(y_pred, tf.float32)
        sigma = y_pred[:, 2] - y_pred[:, 0]
        fvc_pred = y_pred[:, 1]

        # sigma_clip = sigma + C1
        sigma_clip = tf.maximum(sigma, self.C1)
        delta = tf.abs(y_true[:, 0] - fvc_pred)
        delta = tf.minimum(delta, self.C2)
        sq2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))
        metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
        return mean(metric)

    def qloss(self, y_true, y_pred):
        # Pinball loss for multiple quantiles
        qs = [0.2, 0.50, 0.8]
        q = tf.constant(np.array([qs]), dtype=tf.float32)
        e = y_true - y_pred
        v = tf.maximum(q * e, (q - 1) * e)
        return mean(v)

    # =============================#
    def mloss(self, _lambda):
        def loss(y_true, y_pred):
            return _lambda * self.qloss(y_true, y_pred) + (1 - _lambda) * self.score(y_true, y_pred)

        return loss

    def fit(self, dataset, epoch_num=100, print_epoch=10):
        # compile
        self.compile(loss=self.mloss(0.8),
                     metrics=[self.score],
                     optimizer=tf.keras.optimizers.Adam())

        #strategy = tf.distribute.MirroredStrategy()
        #with strategy.scope():

        for epoch in range(epoch_num):
            #print('sTART', epoch)
            for step, (img, x, y, index) in enumerate(dataset.dataset):
                y = tf.reshape(tf.cast(y, tf.float32), (-1, 1))
                with tf.GradientTape() as tape:
                    output = self.call((img, x), training=True)
                    loss = self.loss(y, output)
                    gradients = tape.gradient(loss, self.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                    #print(epoch, step)

            if epoch % print_epoch == 0:
                print('Epoch:', epoch, 'Loss: ', loss.numpy().mean())
                sys.stdout.flush()

    def call(self, inputs, training=False, *args, **kwargs):
        """
        inputs: {imgs: [], info: []}
        """

        imgs = inputs[0]
        info = inputs[1]

        if tf.config.list_physical_devices('GPU'):
            imgs = imgs.gpu()
            info = info.gpu()

        imgs = tf.cast(imgs, tf.float32)
        info = tf.cast(info, tf.float32)

        conv_out = self.conv1(imgs, training=training)
        conv_out = self.flat(conv_out, training=training)

        info = tf.concat((conv_out, info), axis=1)
        out = self.fc(info, training=training)
        out = self.preds([self.p1(out), self.p2(out)], training=training)
        return out

    def inference(self, dataset):
        for step, (img, x, y) in enumerate(dataset.dataset):
            print(x.shape)


if __name__ == '__main__':
    import os

    root_dir = '~/Data/OSIC'
    train_csv_path = os.path.join(root_dir, 'train.csv')
    test_csv_path = os.path.join(root_dir, 'test.csv')
    img_dir = os.path.join(root_dir, 'preprocessing_data')

    train_data = pd.read_csv(train_csv_path)
