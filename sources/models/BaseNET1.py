import tensorflow as tf
import tensorflow.keras.layers as layers


class BaseNET1(tf.keras.Model):

    def __init__(self):
        super(BaseNET1, self).__init__()

        self.downsample = layers.AveragePooling1D(pool_size=10, strides=5)

        self.dilate_1 = layers.Conv1D(filters=8, kernel_size=5, padding="same")
        self.dilate_2 = layers.Conv1D(filters=8, kernel_size=5, padding="same", dilation_rate=2)
        self.dilate_3 = layers.Conv1D(filters=8, kernel_size=5, padding="same", dilation_rate=3)

        self.activate_1 = layers.LeakyReLU(alpha=0.2)
        self.activate_2 = layers.LeakyReLU(alpha=0.2)
        self.activate_3 = layers.LeakyReLU(alpha=0.2)

        self.concat_dilations = layers.Concatenate()
        self.norm = layers.BatchNormalization()

        self.down_conv_1 = layers.Conv1D(filters=16, kernel_size=5, strides=3)
        self.down_activate_1 = layers.LeakyReLU()
        self.down_conv_2 = layers.Conv1D(filters=8, kernel_size=3, strides=3)
        self.down_activate_2 = layers.LeakyReLU()
        self.down_conv_3 = layers.Conv1D(filters=4, kernel_size=3, strides=2)
        self.down_activate_3 = layers.LeakyReLU()

        self.flattener = layers.Flatten()
        self.dense1 = layers.Dense(64)
        self.dense_activate = layers.LeakyReLU()

        self.dense_out = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        down_input = self.downsample(inputs)

        x = self.activate_1(self.dilate_1(down_input))
        y = self.activate_2(self.dilate_2(down_input))
        z = self.activate_3(self.dilate_3(down_input))

        c = self.concat_dilations([x, y, z])
        c = self.norm(c)

        c = self.down_activate_1(self.down_conv_1(c))
        c = self.down_activate_2(self.down_conv_2(c))
        c = self.down_activate_3(self.down_conv_3(c))

        out = self.dense_activate(self.dense1(self.flattener(c)))

        return self.dense_out(out)

    def model(self):
        x = layers.Input(shape=(5000, 2))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
