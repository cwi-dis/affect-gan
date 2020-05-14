import tensorflow as tf
import tensorflow.keras.layers as layers

import config

class ConvLSTM(tf.keras.Model):

    def __init__(self, hparams):
        super(ConvLSTM, self).__init__()

        self.conv_1 = layers.Conv1D(
            filters=hparams[config.HP_FILTERS_CL],
            kernel_size=hparams[config.HP_KERNEL_CL],
            strides=hparams[config.HP_STRIDES_CL],
            padding="same", activation=layers.LeakyReLU())
        self.drop = layers.Dropout(rate=hparams[config.HP_DROPOUT_CL])
        self.avg = layers.AveragePooling1D(
            pool_size=3,
            strides=2,
            padding="same")

        self.lstm_layer = layers.LSTM(units=hparams[config.HP_LSTMCELLS_CL])
        self.dense_output = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        x = self.conv_1(inputs)
        x = self.drop(x, training)
        x = self.avg(x)
        x2 = self.lstm_layer(x)
        return self.dense_output(x2)

    def model(self):
        x = layers.Input(shape=(500, 5))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
