import tensorflow as tf
import tensorflow.keras.layers as layers

import config

class SimpleLSTM(tf.keras.Model):

    def __init__(self, hparams):
        super(SimpleLSTM, self).__init__()

        self.lstm_layer = layers.LSTM(units=hparams[config.HP_CELLS_LSTM])
        self.dense_output = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        x2 = self.lstm_layer(inputs)
        return self.dense_output(x2)

    def model(self):
        x = layers.Input(shape=(500, 5))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))