import config

import tensorflow as tf
import tensorflow.keras.layers as layers

from models.DeepCNN import ChannelDownResLayer

class ChannelCNN(tf.keras.Model):

    def __init__(self, hparams, views):
        super(ChannelCNN, self).__init__()
        self.views = views
        self.layers_count = hparams[config.HP_CDEEP_LAYERS]

        self.channel_down_res_layers = [[
            ChannelDownResLayer(
                channels_out=hparams[config.HP_CDEEP_CHANNELS] * (2 ** l)
            ) for c in range(self.views)
        ] for l in range(self.layers_count - 1)]

        self.channel_down_res_final_layer = [
            ChannelDownResLayer(
                channels_out=hparams[config.HP_CDEEP_CHANNELS] * (2 ** (self.layers_count-1)),
                last_layer=True
            ) for c in range(self.views)
        ]

        self.feature_pool = layers.GlobalAveragePooling1D()
        self.lrelu_out = layers.LeakyReLU()
        self.dense_out = layers.Dense(units=1, activation='sigmoid')


    def call(self, inputs, training=None, mask=None):
        x_c = tf.split(inputs, self.views, -1)
        for layer in range(self.layers_count - 1):
            x_c = [self.channel_down_res_layers[layer][c](x_c[c], training=training) for c in range(self.views)]

        x_c = [self.channel_down_res_final_layer[c](x_c[c], training=training) for c in range(self.views)]

        x = layers.concatenate(x_c)
        x = self.feature_pool(x)
        x = self.lrelu_out(x)
        return self.dense_out(x)


    def model(self):
        x = layers.Input(shape=(500, 5))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
