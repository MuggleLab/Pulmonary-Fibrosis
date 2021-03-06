# Build Model
import sys
import logging

import pandas as pd
import numpy as np

import tensorflow as tf
from tqdm import tqdm

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPool3D, AvgPool3D
from tensorflow.python.keras.layers import Lambda
from tensorflow.keras.backend import mean


class PFPModel(Model):
    def __init__(self):
        super().__init__()
        self.C1, self.C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")
        self.build_model()

    def build_model(self):
        self.conv1 = tf.keras.Sequential([
            Conv3D(filters=10, kernel_size=5, padding='same', activation='relu'),
            # Conv3D(filters=20, kernel_size=5, padding='same', activation='relu'),
            Conv3D(filters=20, kernel_size=3, padding='same', activation='relu'),
            AvgPool3D(pool_size=(3, 3, 3)),
            Conv3D(filters=20, kernel_size=5, padding='same', activation='relu'),
            AvgPool3D(pool_size=(2, 3, 3)),
            Conv3D(filters=10, kernel_size=5, padding='same', activation='relu'),
            Conv3D(filters=10, kernel_size=5, padding='same', activation='relu'),
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

    def fit(self, dataset, epoch_num=100, print_epoch=10, save_epoch=100, save_dir='./'):
        # compile
        self.compile(loss=self.mloss(0.8),
                     metrics=[self.score],
                     optimizer=tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999,
                                                        epsilon=None, decay=0.01, amsgrad=False))

        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():=
        pbar = tqdm(total=epoch_num, file=sys.stdout)
        for epoch in range(1, epoch_num + 1):
            loss_val = []
            for step, (img, x, y, index) in enumerate(dataset.dataset):
                y = tf.reshape(tf.cast(y, tf.float32), (-1, 1))
                with tf.GradientTape() as tape:
                    output = self.call((img, x), training=True)
                    loss = self.loss(y, output)
                    gradients = tape.gradient(loss, self.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                    loss_val.append(loss.numpy())

            # logging.info('Epoch:', epoch, '\tLoss: ', np.mean(loss_val), '\tLast Output: ', output.numpy()[0])
            if epoch % print_epoch == 0 or epoch == epoch_num:
                pbar.write(f'Epoch: {epoch} \tLoss: {np.mean(loss_val)}, \tLast Output: {output.numpy()[0]}')

                # print('Epoch:', epoch, '\tLoss: ', np.mean(loss_val), '\tLast Output: ', output.numpy()[0])
                # sys.stdout.flush()

            if epoch % save_epoch == 0 or epoch == epoch_num:
                loss_mean = np.mean(loss_val)
                loss_mean = int(loss_mean * 100) / 100.0
                self.save_weights(f'{save_dir}/pfp_model_{epoch}_loss({loss_mean})')

            pbar.update(1)

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
        flat_out = self.flat(conv_out, training=training)

        info = tf.concat((flat_out, info), axis=1)
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
