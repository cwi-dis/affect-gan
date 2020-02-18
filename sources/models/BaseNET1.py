import tensorflow as tf
import tensorflow.keras.layers as layers


class BaseNET1(tf.keras.Model):

    def __init__(self):
        super(BaseNET1, self).__init__()

        self.conv_1 = layers.Conv1D(filters=8, kernel_size=5, dilation_rate=2, activation=layers.LeakyReLU())
        self.drop = layers.Dropout(rate=0.5)
        self.avg = layers.AveragePooling1D(pool_size=5, strides=5)
        self.flat = layers.Flatten()
        self.dense = layers.Dense(8, activation=layers.LeakyReLU())
        self.dense_out = layers.Dense(1, activation="sigmoid")


    def call(self, inputs, training=None, mask=None):
        
        x = self.conv_1(inputs)
        x = self.drop(x, training)
        x = self.avg(x)
        x = self.flat(x)
        x = tf.reshape(x, [-1, 784])
        #x = self.dense(x)
        return self.dense_out(x)

    def model(self):
        x = layers.Input(shape=(500, 5))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
