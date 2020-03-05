import config

import tensorflow as tf
import tensorflow.keras.layers as layers
from models.Blocks import *

class ChannelDownResLayer(layers.Layer):
    def __init__(self, channels_out, dropout_rate=0.4, kernel_size=7, w_norm_clip=2, first_layer=False, **kwargs):
        super(ChannelDownResLayer, self).__init__(**kwargs)
        if first_layer:
            self.down_resblock = DownResBlock(channels_out, kernel_size, w_norm_clip, initial_activation=None)
        else:
            self.down_resblock = DownResBlock(channels_out, kernel_size, w_norm_clip)
        self.lrelu = layers.LeakyReLU()
        self.dropout = layers.Dropout(rate=dropout_rate)

    def call(self, inputs, **kwargs):
        x = self.down_resblock(inputs)

        if not self.last_layer:
            x = self.dropout(self.lrelu(x), training=kwargs["training"])

        return x


class DeepCNN(tf.keras.Model):

    def __init__(self, hparams, *args, **kwargs):
        super(DeepCNN, self).__init__(*args, **kwargs)
        self.layers_count = hparams[config.HP_DEEP_LAYERS]
        self.dual_output = hparams[config.HP_LOSS_TYPE] == "DUAL_BCE"
        self.input_len = 500

        self.down_res_layers = [ChannelDownResLayer
            (
                hparams[config.HP_DEEP_CHANNELS] * ((l+3) // 2),
                kernel_size=max(3, hparams[config.HP_DEEP_KERNEL_SIZE] - 2*l)
            ) for l in range(self.layers_count - 1)]
        self.down_res_layer_final = ChannelDownResLayer(
                hparams[config.HP_DEEP_CHANNELS] * ((self.layers_count+2) // 2),
                kernel_size=max(3, hparams[config.HP_DEEP_KERNEL_SIZE] - 2*(self.layers_count-1)),
                last_layer=True)

        self.feature_pool = layers.GlobalAveragePooling1D()
        self.lrelu_out = layers.LeakyReLU()

        if hparams[config.HP_LOSS_TYPE] == "MSE":
            activation = None
        else:
            activation = 'sigmoid'

        self.dense_out = layers.Dense(units=2, activation=activation)

        self.dense_out_a = layers.Dense(units=1, activation=activation, name="arousal_class")
        self.dense_out_v = layers.Dense(units=1, activation=activation, name="valence_class")

    def call(self, inputs, training=None, mask=None):
        x = inputs

        for i in range(self.layers_count - 1):
            x = self.down_res_layers[i](x, training=training)

        x = self.down_res_layer_final(x, training=training)

        x = self.feature_pool(x)
        x = self.lrelu_out(x)

        if self.dual_output:
            return self.dense_out_a(x), self.dense_out_v(x)
        else:
            return self.dense_out_a(x)

    def model(self):
        x = layers.Input(shape=(500, 5))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
