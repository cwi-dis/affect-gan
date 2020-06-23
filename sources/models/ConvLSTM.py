import tensorflow as tf
import tensorflow.keras.layers as layers

import config

class ConvLSTM(tf.keras.Model):

    def __init__(self, hparams):
        super(ConvLSTM, self).__init__()

        self.expand = layers.Conv1D(
            filters=6,
            kernel_size=6,
            strides=2,
            padding="same"
        )

        self.lstm_layer = layers.LSTM(
            units=6,
            dropout=0.4
        )

        self.dense_output = layers.Dense(2, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        x = self.expand(inputs)
        x = self.lstm_layer(x, training=training)
        return self.dense_output(x)

    def model(self):
        x = layers.Input(shape=(500, 5))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
