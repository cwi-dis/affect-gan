import tensorflow as tf
import tensorflow.keras.layers as layers


class SimpleLSTM(tf.keras.Model):

    def __init__(self):
        super(SimpleLSTM, self).__init__()

        self.norm = layers.BatchNormalization(axis=-1)
        self.lstm_layer = layers.LSTM(units=16)
        self.dense_output = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        x = self.norm(inputs)
        x2 = self.lstm_layer(x)
        return self.dense_output(x2)

    def model(self):
        x = layers.Input(shape=(5000, 2))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))