import config

import tensorflow as tf
import tensorflow.keras.layers as layers


class ChannelDownResBlock(tf.keras.layers.Layer):
    def __init__(self, channels, activation=None):
        super(ChannelDownResBlock, self).__init__()
        self.out_channels = channels
        self.conv1 = layers.Conv1D(filters=channels, kernel_size=3, padding="same", activation=layers.LeakyReLU(alpha=0.2))
        self.conv2 = layers.Conv1D(filters=channels, kernel_size=3, padding="same")

        self.pool = layers.AveragePooling1D(pool_size=2, strides=2)

        self.shortcut_conv = layers.Conv1D(filters=channels, kernel_size=1, padding="same")
        self.shortcut_pool = layers.AveragePooling1D(pool_size=2, strides=2)

        self.out_activation = activation

    def call(self, inputs, **kwargs):
        input_channels = inputs.shape.as_list()[-1]

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool(x)

        if self.out_channels != input_channels:
            inputs = self.shortcut_conv(inputs)
        inputs = self.shortcut_pool(inputs)

        out = x + inputs

        if self.out_activation is not None:
            out = self.out_activation(out)

        return out


class DeepCNN(tf.keras.Model):

    def __init__(self, hparams, *args, **kwargs):
        super(DeepCNN, self).__init__(*args, **kwargs)
        self.layers_count = hparams[config.HP_DEEP_LAYERS]

        self.down_resblock0 = ChannelDownResBlock(hparams[config.HP_DEEP_CHANNELS])

        self.lrelu1 = layers.LeakyReLU()
        self.drop1 = layers.Dropout(rate=0.4)
        self.down_resblock1 = ChannelDownResBlock(hparams[config.HP_DEEP_CHANNELS] * 2)

        self.lrelu2 = layers.LeakyReLU()
        self.drop2 = layers.Dropout(rate=0.4)
        self.down_resblock2 = ChannelDownResBlock(hparams[config.HP_DEEP_CHANNELS] * 4)

        self.feature_pool = layers.GlobalAveragePooling1D()
        self.lrelu_out = layers.LeakyReLU()
        self.dense_out = layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        x = self.down_resblock0(inputs)

        if self.layers_count != 1:
            x = self.lrelu1(x)
            x = self.drop1(x, training=training)
            x = self.down_resblock1(x)

        if self.layers_count == 3:
            x = self.lrelu2(x)
            x = self.drop2(x, training=training)
            x = self.down_resblock2(x)

        x = self.feature_pool(x)
        x = self.lrelu_out(x)

        return self.dense_out(x)

    def model(self):
        x = layers.Input(shape=(500, 5))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
