import tensorflow as tf
import tensorflow.keras.layers as layers

import config
from models.Blocks import *

class Generator(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super(Generator, self).__init__(*args, **kwargs)

        self.up_0 = UpResLayer(channels_out=8)
        self.non_local = AttentionLayer(filters=8, use_actnormdrop=True)
        self.up_1 = UpResLayer(channels_out=6, use_actnormdrop=True)
        self.final_conv = layers.Conv1D(filters=5, kernel_size=3, padding="same",
                                        kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1, axis=[0, 1]))

    def call(self, inputs, training=None, mask=None):
        x = tf.expand_dims(inputs, axis=-1)
        x = self.up_0(x)
        x = self.non_local(x)
        x = self.up_1(x)
        x = self.final_conv(x)
        x = tf.keras.activations.tanh(x)
        return x

    def model(self):
        x = layers.Input(shape=(125, 1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class Discriminator(tf.keras.Model):
    def __init__(self, hparams, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)

        self.num_layers = hparams[config.HP_ATT2_LAYERS]

        self.att_layers = []
        for l in range(self.num_layers):
            self.att_layers.append(AttentionLayer(
                filters=hparams[config.HP_ATT2_FILTERS],
                use_actnormdrop=l != self.num_layers-1
            ))

        self.avg = layers.GlobalAveragePooling1D()

        self.dense_output = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in range(self.num_layers):
            x = self.att_layers[layer](x, training=training)

        x = self.avg(x)
        return self.dense_output(x)

    def model(self):
        x = layers.Input(shape=(500, 5))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
